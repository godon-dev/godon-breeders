1|#
2|# Copyright (c) 2019 Matthias Tafelmeier.
3|#
4|# Tests for the DetectionCoordinator state machine.
5|#
6|
7|import pytest
8|import sys
9|import os
10|from unittest.mock import MagicMock, patch
11|
12|sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))
13|
14|# Mock heavy imports
15|sys.modules['wmill'] = MagicMock()
16|sys.modules['psycopg2'] = MagicMock()
17|sys.modules['optuna'] = MagicMock()
18|sys.modules['optuna.storages'] = MagicMock()
19|sys.modules['optuna.trial'] = MagicMock()
20|from optuna.trial import TrialState
21|sys.modules['optuna.samplers'] = MagicMock()
22|sys.modules['prometheus_api_client'] = MagicMock()
23|sys.modules['prometheus_client'] = MagicMock()
24|sys.modules['psycopg2'] = MagicMock()
25|
26|from engine.detection_coordinator import DetectionCoordinator
27|
28|
29|def _base_config():
30|    return {
31|        'breeder': {'uuid': 'test-uuid', 'name': 'test', 'type': 'greenhouse'},
32|        'objectives': [{'name': 'growth', 'direction': 'maximize'}],
33|        'detection': {'warmup_trials': 3, 'push_block_size': 3, 'pause_block_size': 3, 'recover_trials': 2},
34|    }
35|
36|
37|def _mock_db(return_value=True):
38|    """Mock _with_shared_db that returns a fixed value."""
39|    db = MagicMock(return_value=return_value)
40|    return db
41|
42|
43|def _mock_study(n_complete=0):
44|    """Create a mock study with n_complete complete trials."""
45|    study = MagicMock()
46|    trials = []
47|    for i in range(n_complete):
48|        t = MagicMock()
49|        t.state = TrialState.COMPLETE
50|        t.values = [0.5 + i * 0.01]
51|        t.user_attrs = {'effectuation_params': '{"heating": 20.0, "light": 300}'}
52|        trials.append(t)
53|    for i in range(5):
54|        t = MagicMock()
55|        t.state = TrialState.FAIL
56|        t.user_attrs = {}
57|        trials.append(t)
58|    study.trials = trials
59|    return study
60|
61|
62|def _create_coordinator(breeder_id='test-uuid', n_complete=0, db_return=True):
63|    config = _base_config()
64|    db = _mock_db(db_return)
65|    collect_fn = MagicMock(return_value=[
66|        {'name': 'heating', 'upper': 30, 'lower': 10, 'range': 20, 'is_int': False},
67|        {'name': 'light', 'upper': 1000, 'lower': 0, 'range': 1000, 'is_int': False},
68|    ])
69|    coord = DetectionCoordinator(
70|        breeder_id=breeder_id,
71|        config=config,
72|        shared_db_fn=db,
73|        collect_upper_bounds_fn=collect_fn,
74|    )
75|    return coord, db
76|
77|
78|class TestWarmup:
79|    def test_returns_optimize_during_warmup(self):
80|        coord, _ = _create_coordinator()
81|        study = _mock_study(n_complete=1)  # Less than warmup_target=3
82|        trial = MagicMock()
83|        with patch.object(coord, '_any_active_round', return_value=False), \
84|             patch.object(coord, '_count_complete_trials_db', return_value=-1):
85|            decision = coord.decide_trial(trial, study)
86|        assert decision['mode'] == 'optimize'
87|        assert decision['params'] is None
88|        assert coord.state == DetectionCoordinator.WARMUP
89|
90|    def test_transitions_to_sender_after_warmup(self):
91|        coord, db = _create_coordinator()
92|        study = _mock_study(n_complete=3)  # Equals warmup_target
93|        trial = MagicMock()
94|        # db returns True for try_start_round
95|        with patch.object(coord, '_any_active_round', return_value=False), \
96|             patch.object(coord, '_count_complete_trials_db', return_value=-1):
97|            decision = coord.decide_trial(trial, study)
98|        assert decision['mode'] == 'optimize'  # Last warmup trial
99|        assert coord.state == DetectionCoordinator.SENDER_PUSH
100|
101|    def test_transitions_to_receiver_if_cannot_start(self):
102|        coord, db = _create_coordinator(db_return=False)
103|        study = _mock_study(n_complete=3)
104|        trial = MagicMock()
105|        # Need _any_active_round to return True
106|        with patch.object(coord, '_any_active_round', return_value=True):
107|            decision = coord.decide_trial(trial, study)
108|        assert coord.state == DetectionCoordinator.RECEIVER_HOLD
109|
110|
111|class TestSenderPing:
112|    def test_ping_returns_impulse_with_extreme_params(self):
113|        coord, _ = _create_coordinator()
114|        coord.state = DetectionCoordinator.SENDER_PUSH
115|        coord._baseline_params = {'heating': 20.0, 'light': 300.0}
116|        trial = MagicMock()
117|        decision = coord.decide_trial(trial, MagicMock())
118|        assert decision['mode'] == 'impulse'
119|        assert decision['impulse_phase'] == 'ping'
120|        assert decision['params'] is not None
121|        # Heating should be at upper bound (30 * 1.0)
122|        assert decision['params']['heating'] == 30.0
123|
124|    def test_ping_increments_counter(self):
125|        coord, _ = _create_coordinator()
126|        coord.state = DetectionCoordinator.SENDER_PUSH
127|        coord._baseline_params = {'heating': 20.0}
128|        trial = MagicMock()
129|        coord.decide_trial(trial, MagicMock())
130|        assert coord._push_count == 1
131|
132|    def test_ping_transitions_to_listen(self):
133|        coord, _ = _create_coordinator()
134|        coord.state = DetectionCoordinator.SENDER_PUSH
135|        coord._baseline_params = {'heating': 20.0}
136|        trial = MagicMock()
137|        coord.decide_trial(trial, MagicMock())
138|        assert coord.state == DetectionCoordinator.SENDER_PAUSE
139|
140|
141|class TestSenderListen:
142|    def test_listen_returns_baseline_params(self):
143|        coord, _ = _create_coordinator()
144|        coord.state = DetectionCoordinator.SENDER_PAUSE
145|        coord._baseline_params = {'heating': 20.0, 'light': 300.0}
146|        trial = MagicMock()
147|        decision = coord.decide_trial(trial, MagicMock())
148|        assert decision['mode'] == 'impulse'
149|        assert decision['impulse_phase'] == 'listen'
150|        assert decision['params']['heating'] == 20.0  # Baseline, not extreme
151|
152|    def test_listen_transitions_back_to_ping_if_round_not_done(self):
153|        coord, _ = _create_coordinator()
154|        coord.state = DetectionCoordinator.SENDER_PAUSE
155|        coord._push_count = 1  # Less than impulses_per_round=3
156|        coord._baseline_params = {'heating': 20.0}
157|        coord.decide_trial(MagicMock(), MagicMock())
158|        assert coord.state == DetectionCoordinator.SENDER_PUSH
159|
160|    def test_listen_transitions_to_done_if_round_complete(self):
161|        coord, _ = _create_coordinator()
162|        coord.state = DetectionCoordinator.SENDER_PAUSE
163|        coord._push_count = 3  # Equals impulses_per_round
164|        coord._baseline_params = {'heating': 20.0}
165|        coord.decide_trial(MagicMock(), MagicMock())
166|        assert coord.state == DetectionCoordinator.SENDER_DONE
167|
168|
169|class TestSenderDone:
170|    def test_completes_round_and_enters_recover(self):
171|        coord, _ = _create_coordinator()
172|        coord.state = DetectionCoordinator.SENDER_DONE
173|        decision = coord.decide_trial(MagicMock(), MagicMock())
174|        assert decision['mode'] == 'optimize'
175|        assert coord.state == DetectionCoordinator.RECOVER
176|        assert coord._recover_count == 0
177|
178|
179|class TestRecover:
180|    def test_optimizes_during_recovery(self):
181|        coord, _ = _create_coordinator()
182|        coord.state = DetectionCoordinator.RECOVER
183|        coord._recover_count = 0
184|        decision = coord.decide_trial(MagicMock(), MagicMock())
185|        assert decision['mode'] == 'optimize'
186|        assert coord._recover_count == 1
187|
188|    def test_becomes_sender_after_recovery(self):
189|        coord, db = _create_coordinator()
190|        coord.state = DetectionCoordinator.RECOVER
191|        coord._recover_count = 1  # One more and it's done (target=2)
192|        coord._baseline_params = {'heating': 20.0}
193|        study = _mock_study(n_complete=5)
194|        decision = coord.decide_trial(MagicMock(), study)
195|        assert coord.state == DetectionCoordinator.SENDER_PUSH
196|        assert coord._push_count == 0
197|
198|
199|class TestReceiverHold:
200|    def test_returns_hold_with_baseline_params(self):
201|        coord, _ = _create_coordinator()
202|        coord.state = DetectionCoordinator.RECEIVER_HOLD
203|        coord._baseline_params = {'heating': 20.0}
204|        decision = coord.decide_trial(MagicMock(), MagicMock())
205|        assert decision['mode'] == 'hold'
206|        assert decision['params']['heating'] == 20.0
207|
208|    def test_enters_recover_when_sender_finishes(self):
209|        coord, _ = _create_coordinator()
210|        coord.state = DetectionCoordinator.RECEIVER_HOLD
211|        coord._baseline_params = {'heating': 20.0}
212|        with patch.object(coord, '_any_active_round', return_value=False):
213|            decision = coord.decide_trial(MagicMock(), MagicMock())
214|        assert coord.state == DetectionCoordinator.RECOVER
215|
216|
217|class TestGuardrailFail:
218|    def test_ping_fail_triggers_aimd_backoff(self):
219|        coord, _ = _create_coordinator()
220|        coord.state = DetectionCoordinator.SENDER_PUSH
221|        coord._push_count = 2
222|        coord._baseline_params = {'heating': 20.0}
223|        # Generate impulse params first
224|        coord._get_impulse_params()
225|        original_scale = coord._impulse_scale
226|        coord.on_guardrail_fail({'heating': 30.0})
227|        assert coord._impulse_scale == original_scale * 0.5
228|        assert coord._push_count == 1  # Decremented
229|
230|    def test_listen_fail_does_not_trigger_backoff(self):
231|        coord, _ = _create_coordinator()
232|        coord.state = DetectionCoordinator.SENDER_PAUSE
233|        coord._impulse_scale = 1.0
234|        coord.on_guardrail_fail({'heating': 20.0})
235|        assert coord._impulse_scale == 1.0  # Unchanged
236|
237|
238|class TestStateCleanup:
239|    def test_cleans_up_stale_rounds_on_init(self):
240|        coord, db = _create_coordinator()
241|        trial = MagicMock()
242|        study = _mock_study(n_complete=0)
243|        with patch.object(coord, '_any_active_round', return_value=False), \
244|             patch.object(coord, '_count_complete_trials_db', return_value=-1):
245|            coord.decide_trial(trial, study)
246|        # Should have called db at least once for cleanup
247|        assert db.call_count >= 1
248|