# NeuralFSM: Complete Algorithm Description

## Algorithm 1: NeuralFSM Training Algorithm

```pseudocode
Algorithm: NeuralFSM Training
Require: 
    - Dataset D = {(q_i, a_i)}_{i=1}^N  // (question, answer) pairs, unified format across domains
    - domain d ∈ {mmlu, gsm8k, humaneval, hotpotqa, alfworld, math}  // dataset identifier
    - hyperparameters α, β, γ, δ, λ, epochs, batch_size, lr
Ensure: Trained TGN model θ*

// Note: 
// - D format: Each dataset (MMLU, GSM8K, etc.) has different raw formats but is unified to (q, a) pairs
// - domain d: Determines FSM structure (agent roles, states), prompt templates, answer validation

1. Initialize Total FSM F = (S, A, δ, L, C, s₀)
2. Initialize TGN model TGN_θ with question encoder
3. Initialize optimizer OPT (Adam, lr)
4. Initialize embedding model E (SentenceTransformer)
5. Split D into D_train, D_test
6. Create batches B_train from D_train
7. 
8. for epoch = 1 to epochs do
9.     for each batch B in B_train do
10.        for each question q in B do
11.            q_emb ← E.encode(q)  // Task embedding
12.            h_a ← GetAgentFeatures(topology)
13.            h_c ← GetContextFeatures(d)
14.            
15.            // TGN forward pass (task-adaptive)
16.            fsm_out ← TGN_θ(h_a, h_c, s₀, q_emb)
17.            // Output: P(·|s, c, q_emb), w(s, a, q_emb)
18.            
19.            // Execute FSM with probabilistic sampling
20.            fsm_mgr.reset()
21.            (ans, log) ← ExecuteFSM(topology, fsm_mgr, fsm_out, q, TGN_θ, q_emb, h_a, h_c)
22.            
23.            // Evaluate and compute reward
24.            R ← 1.0 if CheckCorrect(ans, gt) else 0.0
25.            
26.            // Compute losses
27.            L_policy ← -log P(τ|θ, q_emb) · R  // Policy gradient loss
28.            
29.            targets_t ← GetTransitionTargets(fsm_mgr.history)
30.            L_transition ← 0
31.            if targets_t ≠ ∅ then
32.                L_transition ← -mean(log P(s_t|s_{t-1}, c, q_emb, θ) for s_t in targets_t)  // State transition loss
33.            end if
34.            
35.            targets_l ← GetListenerTargets(fsm_mgr.listeners)
36.            L_listener ← BCE(w(s, a, q_emb), targets_l)  // Listener path loss
37.            
38.            C_episode ← ComputeCost(log)
39.            L_cost ← (C_episode - C_baseline)²  // LLM cost loss
40.            
41.            if enable_protect then
42.                L_protection ← Σ_{(u,v)∈E} [risk(u,v) × ||m(u→v)||²]  // Protection loss
43.            else
44.                L_protection ← 0
45.            end if
46.            
47.            // Total loss
48.            L_total ← α·L_policy + β·L_transition + γ·L_listener + δ·L_cost + λ·L_protection
49.            
50.            // Update parameters
51.            θ ← θ - lr · ∇_θ L_total
52.        end for
53.    end for
54.end for
55.
56.return TGN_θ*
```

## Algorithm 2: Task-Adaptive FSM Execution Algorithm

```pseudocode
Algorithm: ExecuteFSM (Task-Adaptive FSM Execution)
Require: topology, fsm_mgr, fsm_out, q, TGN_θ, q_emb, h_a, h_c, T_max
Ensure: final_answer, execution_log

1. fsm_mgr.reset()
2. s_t ← s₀
3. t ← 0
4. log_prob ← 0
5. 
6. while t < T_max do
7.     s_curr ← fsm_mgr.get_state(s_t)
8.     
9.     // Dynamic recompute (task-adaptive)
10.    if TGN_θ ≠ null and q_emb ≠ null then
11.        fsm_out ← TGN_θ(h_a, h_c, s_t, q_emb)
12.    end if
13.    P_trans ← fsm_out['transition_probs']  // P(·|s_t, c, q_emb)
14.    
15.    // Probabilistic sampling
16.    s_{t+1} ~ Multinomial(P_trans)
17.    log_prob ← log_prob + log P(s_{t+1}|s_t, c, q_emb, θ)
18.    
19.    // Match from Total FSM
20.    cond ← fsm_mgr.get_condition(s_t, s_{t+1})
21.    
22.    // Execute agent
23.    agent_id ← s_curr.responsible_agent_id
24.    agent ← topology.get_agent(agent_id)
25.    out ← agent.execute(q, context)
26.    
27.    // Check completion
28.    if CheckComplete(s_curr, out) then
29.        if s_curr.is_final then
30.            ans ← ExtractAnswer(out)
31.            break
32.        end if
33.    end if
34.    
35.    // Sample listeners (task-adaptive)
36.    w_listen ← fsm_out['listener_weights']  // w(s, a, q_emb)
37.    listeners ← SampleListeners(w_listen, k=3)
38.    
39.    // Propagate messages
40.    for each lid in listeners do
41.        agent ← topology.get_agent(lid)
42.        msg ← FormatMsg(out, s_curr)
43.        agent.add_msg(msg)
44.    end for
45.    
46.    // Update FSM
47.    fsm_mgr.transition(s_{t+1})
48.    fsm_mgr.set_listeners(s_t, listeners)
49.    fsm_mgr.history.append((s_t, s_{t+1}))
50.    
51.    s_t ← s_{t+1}
52.    t ← t + 1
53.end while
54.
55.log ← {log_prob, fsm_mgr.history, fsm_mgr.listeners, ans}
56.return (ans, log)
```

## Algorithm 3: Protection Mechanism Integration Algorithm

```pseudocode
Algorithm: ProtectedTGN Forward Pass
Require: h_a ∈ ℝ^{N×d}, E ∈ ℝ^{2×M}, q_emb ∈ ℝ³⁸⁴, s_t, h_c, graph G, counts, embeddings
Ensure: protected_outputs

1. // Compute protection priorities (static)
2. π ← ComputePriorities(G)  // π[i] = w_BC·BC(i) + w_PR·PR(i)
3.
4. // Compute anomaly scores (dynamic)
5. α ← ComputeAnomaly(counts, embeddings)  // α[i] = λ_f·α_f(i) + λ_s·α_s(i)
6.
7. // Compute trust scores
8. for each node i do
9.     trust[i] ← (1 - α[i]) · (1 + π[i])
10.end for
11.
12.// Compute message weights (learnable MLP)
13.for each edge (u, v) in E do
14.    w[u,v] ← MLP(trust[u], π[v])  // Input: [trust_source, priority_target]
15.end for
16.
17.// Training-time defense: filter low-trust connections
18.(E_prot, w_prot) ← FilterEdges(E, w, threshold=0.3)
19.
20.// Call TGN with question embedding
21.fsm_out ← TGN_θ(h_a, E_prot, s_t, h_c, q_emb, history)
22.
23.// Runtime defense: message attenuation
24.if E_prot.size == E.size then
25.    fsm_out['h_a'] ← Attenuate(fsm_out['h_a'], h_a, E, w, α)
26.    // out = trust × TGN_out + (1-trust) × h_a
27.end if
28.
29.return fsm_out
```
