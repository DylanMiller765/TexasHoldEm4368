import sys
import os
from flask import Flask, jsonify, request, send_from_directory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from src.game_engine.card import Card
from src.game_engine.game import Game, GameState, Action, Street
from src.game_engine.hand_eval import evaluate
from src.policy_network.network import PolicyNet
from src.policy_network.agent import PokerAgent
from src.state_encoder.encoder import encode_state
from src.evaluation.evaluate import Evaluator

app = Flask(__name__)
_evaluator = Evaluator()

ASSETS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "assets")
TEMPLATES_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "templates")

HUMAN_ID = 0
AI_ID = 1

session = \
{
    "game":          None,
    "human_chips":   1000,
    "ai_chips":      1000,
    "iterator":      None,
    "hand_over":     False,
    "community":     [],
    "street":        "preflop",
    "pending_human": False,
    "pending_state": None,
    "last_payload":  None,
}

MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models', 'poker_model_latest.pt')
_model = PolicyNet()
_agent = PokerAgent(_model, encode_state)

if os.path.exists(MODEL_PATH):
    _agent.load(MODEL_PATH)
    print(f'  Loaded AI model from {MODEL_PATH}')
else:
    print(f'  WARNING: No model found at {MODEL_PATH}. AI will use random weights.')

def ai_action(state: GameState):
    _agent.reset()
    return _agent.select_action(state)

def card_to_str(card: Card) -> str:
    return repr(card)

def hand_name(hole_cards, community_cards) -> str:
    if len(community_cards) < 3:
        return ""
    try:
        hr = evaluate(hole_cards, community_cards)
        return hr.category.name.replace("_", " ").title()
    except Exception:
        return ""

def build_payload(g, actor, action, amount, hand_over=False, winner=None, showdown=False, pending_human=False):
    state = g.get_state_for_player(HUMAN_ID)
    to_call = max(0, state.current_bet - state.player_round_bet)
    can_check = to_call == 0

    ai_hn = hand_name(g.players[AI_ID].hole_cards,    g.community_cards) if len(g.community_cards) >= 3 else None
    human_hn = hand_name(g.players[HUMAN_ID].hole_cards, g.community_cards) if len(g.community_cards) >= 3 else None

    return \
    {
        "actor":             actor,
        "action":            action.name.lower() if action else None,
        "amount":            amount,
        "human_chips":       g.players[HUMAN_ID].chips,
        "ai_chips":          g.players[AI_ID].chips,
        "pot":               g.pot.total,
        "street":            g.street.value,
        "community":         [card_to_str(c) for c in g.community_cards],
        "hand_over":         hand_over,
        "winner":            winner,
        "showdown":          showdown,
        "ai_hand_name":      ai_hn,
        "human_hand_name":   human_hn,
        "human_hole":        [card_to_str(c) for c in g.players[HUMAN_ID].hole_cards],
        "ai_hole":           [card_to_str(c) for c in g.players[AI_ID].hole_cards] if (showdown or hand_over) else None,
        "pending_human":     pending_human,
        "to_call":           to_call,
        "can_check":         can_check,
        "min_raise":         state.min_raise,
        "max_raise":         state.max_raise,
    }

def hand_iterator(g):
    streets = \
    [
        (Street.PREFLOP, 0),
        (Street.FLOP,    3),
        (Street.TURN,    1),
        (Street.RIVER,   1),
    ]

    for idx, (street, n_cards) in enumerate(streets):
        g.street = street
        if idx > 0:
            g.betting_history.append([])
            if not g.hand_over:
                g._deal_community(n_cards)
                yield build_payload(g, None, None, 0)

        if g.hand_over:
            break

        if g._can_act_count() < 2:
            g.pot.new_round()
            continue

        first = g.dealer if street == Street.PREFLOP else 1 - g.dealer
        actors = [first, 1 - first]
        i = 0

        while i < len(actors):
            pid = actors[i]
            p = g.players[pid]
            opp = g.players[1 - pid]

            if p.is_folded or p.is_all_in:
                i += 1; continue
            if opp.is_folded:
                break
            if opp.is_all_in:
                to_call = (g.pot.player_round_bet(opp.player_id) - g.pot.player_round_bet(pid))
                if to_call <= 0:
                    i += 1; continue

            state = g.get_state_for_player(pid)

            if pid == HUMAN_ID:
                session["pending_human"] = True
                session["pending_state"] = state
                yield build_payload(g, "human", None, 0, pending_human=True)
                raw_action, raw_amount = session.pop("human_action")
                session["pending_human"] = False
            else:
                raw_action, raw_amount = ai_action(state)

            action, amount = g._validate_action(pid, raw_action, raw_amount)
            g._apply_action(pid, action, amount)
            g.betting_history[-1].append((pid, action, amount))

            yield build_payload(g, "human" if pid == HUMAN_ID else "ai", action, amount)

            if action == Action.FOLD:
                g.hand_over = True
                g.winner_ids = [1 - pid]
                break

            if action in (Action.RAISE, Action.ALL_IN):
                if not opp.is_all_in and not opp.is_folded:
                    if i + 1 >= len(actors) or actors[i + 1] != (1 - pid):
                        actors.append(1 - pid)

            i += 1

        g.pot.new_round()
        if g.hand_over:
            break

    result = g._resolve()
    showdown = len(g.community_cards) == 5 and not any(p.is_folded for p in g.players)
    winners = result.get("winners", [])

    if HUMAN_ID in winners and AI_ID in winners:
        winner_str = "tie"
    elif HUMAN_ID in winners:
        winner_str = "human"
    else:
        winner_str = "ai"

    session["human_chips"] = g.players[HUMAN_ID].chips
    session["ai_chips"] = g.players[AI_ID].chips
    session["hand_over"] = True

    _evaluator.record(winner_str, g.players[HUMAN_ID].chips, g.players[AI_ID].chips)

    yield build_payload(g, None, None, 0,hand_over=True, winner=winner_str,showdown=showdown)


@app.route("/")
def index():
    return send_from_directory(TEMPLATES_DIR, "index.html")

@app.route("/cards/<path:filename>")
def card_image(filename):
    return send_from_directory(ASSETS_DIR, filename)

@app.route("/greenFelt.jpeg")
def green_felt():
    return send_from_directory(ASSETS_DIR, "greenFelt.jpeg")

@app.route("/flag.png")
def flag_image():
    return send_from_directory(ASSETS_DIR, "flag.png")

@app.route("/cowboy_left.png")
def cowboy_left():
    return send_from_directory(ASSETS_DIR, "cowboy_left.png")

@app.route("/cowboy_right.png")
def cowboy_right():
    return send_from_directory(ASSETS_DIR, "cowboy_right.png")

@app.route("/new_hand", methods=["POST"])
def new_hand():
    g = Game\
    (
        stacks=(session["human_chips"], session["ai_chips"]),
        small_blind=10,
        big_blind=20,
    )
    g._start_hand()
    g._post_blinds()
    g.betting_history.append([])

    session["game"] = g
    session["hand_over"] = False
    session["community"] = []
    session["street"] = "preflop"
    session["pending_human"] = False
    session["pending_state"] = None
    session["iterator"] = hand_iterator(g)

    state   = g.get_state_for_player(HUMAN_ID)
    to_call = max(0, state.current_bet - state.player_round_bet)

    return jsonify\
    (
    {
        "human_hole":    [card_to_str(c) for c in g.players[HUMAN_ID].hole_cards],
        "ai_hole":       None,
        "community":     [],
        "human_chips":   g.players[HUMAN_ID].chips,
        "ai_chips":      g.players[AI_ID].chips,
        "pot":           g.pot.total,
        "street":        "preflop",
        "to_call":       to_call,
        "can_check":     to_call == 0,
        "min_raise":     state.min_raise,
        "max_raise":     state.max_raise,
        "pending_human": False,
    }
    )

@app.route("/step", methods=["POST"])
def step():
    it = session.get("iterator")
    if it is None:
        return jsonify({"error": "No active hand."}), 400
    if session.get("pending_human"):
        return jsonify({"error": "Waiting for human action."}), 400
    try:
        payload = next(it)
        session["last_payload"] = payload
        return jsonify(payload)
    except StopIteration:
        return jsonify({"error": "Hand already finished."}), 400

@app.route("/action", methods=["POST"])
def player_action():
    if not session.get("pending_human"):
        return jsonify({"error": "Not waiting for human action."}), 400

    data = request.get_json()
    action_str = data.get("action", "").lower()
    amount = int(data.get("amount", 0))

    action_map =\
    {
        "fold":   Action.FOLD,
        "check":  Action.CHECK,
        "call":   Action.CALL,
        "raise":  Action.RAISE,
        "all_in": Action.ALL_IN,
    }
    action = action_map.get(action_str)
    if action is None:
        return jsonify({"error": f"Unknown action: {action_str}"}), 400

    session["human_action"]  = (action, amount)
    session["pending_human"] = False

    it = session.get("iterator")
    try:
        payload = next(it)
        session["last_payload"] = payload
        return jsonify(payload)
    except StopIteration:
        return jsonify({"error": "Hand finished unexpectedly."}), 400

@app.route("/stats", methods=["GET"])
def stats():
    return jsonify(_evaluator.summary())

if __name__ == "__main__":
    print("\n  Texas Hold'em server starting...")
    print("  Open http://localhost:5000 in your browser\n")
    app.run(debug=False, port=5000)
