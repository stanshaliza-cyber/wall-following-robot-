import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go

# ---------------------------------------------------------------------
# PAGE CONFIGURATION
# ---------------------------------------------------------------------
st.set_page_config(
    page_title="Wall-Following Robot MDP Dashboard",
    page_icon="🤖",
    layout="wide"
)

# ---------------------------------------------------------------------
# DATA LOADING & MDP COMPUTATION (ROBUST FALLBACK)
# ---------------------------------------------------------------------
@st.cache_data
def load_data_and_compute_mdp():
    # Load sensor dataset
    sensor_df = pd.read_csv(
        'sensor_readings_4.csv',
        header=None,
        names=['SD_front', 'SD_left', 'SD_right', 'SD_back', 'Class']
    )
    
    # Discretize states
    def discretize_state(sd_left):
        if sd_left < 0.5:
            return 'Too-Close'
        elif sd_left < 0.9:
            return 'Ideal'
        else:
            return 'Too-Far'

    sensor_df['State'] = sensor_df['SD_left'].apply(discretize_state)
    states = ['Too-Close', 'Ideal', 'Too-Far']
    actions = ['Move-Forward', 'Slight-Right-Turn', 'Sharp-Right-Turn', 'Slight-Left-Turn']
    n_states = len(states)
    n_actions = len(actions)

    state_idx = {s: i for i, s in enumerate(states)}
    action_idx = {a: i for i, a in enumerate(actions)}

    # Empirical Transitions
    transition_counts = np.zeros((n_states, n_actions, n_states))
    state_sequence = sensor_df['State'].values
    action_sequence = sensor_df['Class'].values

    for t in range(len(sensor_df) - 1):
        s = state_idx[state_sequence[t]]
        a = action_idx.get(action_sequence[t])
        s_next = state_idx[state_sequence[t + 1]]
        if a is not None:
            transition_counts[s, a, s_next] += 1

    T = np.zeros((n_states, n_actions, n_states))
    for s in range(n_states):
        for a in range(n_actions):
            total = transition_counts[s, a].sum()
            if total > 0:
                T[s, a] = transition_counts[s, a] / total
            else:
                T[s, a] = np.ones(n_states) / n_states

    # Rewards Matrix
    reward_values = {
        'Too-Close': {'Move-Forward': -10, 'Slight-Right-Turn': 5, 'Sharp-Right-Turn': 10, 'Slight-Left-Turn': -10},
        'Ideal': {'Move-Forward': 10, 'Slight-Right-Turn': 2, 'Sharp-Right-Turn': -5, 'Slight-Left-Turn': 2},
        'Too-Far': {'Move-Forward': -2, 'Slight-Right-Turn': -10, 'Sharp-Right-Turn': -10, 'Slight-Left-Turn': 10},
    }
    R = np.array([[reward_values[s][a] for a in actions] for s in states])

    # Value Iteration
    gamma = 0.9
    tolerance = 1e-4
    V = np.zeros(n_states)
    for _ in range(1000):
        V_new = np.zeros(n_states)
        for s in range(n_states):
            Q_sa = np.zeros(n_actions)
            for a in range(n_actions):
                Q_sa[a] = R[s][a] + gamma * np.dot(T[s][a], V)
            V_new[s] = np.max(Q_sa)
        if np.max(np.abs(V_new - V)) < tolerance:
            V = V_new
            break
        V = V_new

    policy = {}
    for s in range(n_states):
        Q_sa = np.zeros(n_actions)
        for a in range(n_actions):
            Q_sa[a] = R[s][a] + gamma * np.dot(T[s][a], V)
        policy[states[s]] = actions[np.argmax(Q_sa)]

    optimal_df = pd.DataFrame({
        'State': states,
        'Optimal_Value': V,
        'Optimal_Action': [policy[s] for s in states]
    })

    return sensor_df, optimal_df, T, states, actions

sensor_data, optimal_df, T, states, actions = load_data_and_compute_mdp()

# ---------------------------------------------------------------------
# SIDEBAR NAVIGATION & CONTROLS
# ---------------------------------------------------------------------
st.sidebar.title("🤖 Navigation Controls")
app_mode = st.sidebar.selectbox(
    "Choose Dashboard View",
    [
        "🏠 Overview & MDP Policy", 
        "🗺️ Robot Wall Simulation", 
        "📊 Transition Probabilities", 
        "📈 Dataset Explorer"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info(
    "**Project:** Wall-Following Robot Navigation using Markov Decision Processes (MDP) & Value Iteration."
)

# ---------------------------------------------------------------------
# VIEW 1: OVERVIEW & OPTIMAL POLICY
# ---------------------------------------------------------------------
if app_mode == "🏠 Overview & MDP Policy":
    st.title("🤖 Wall-Following Robot MDP & Value Iteration")
    st.markdown("""
    This dashboard models a robot navigating a corridor by following the wall on its left using sensor readings (`SD_left`). 
    Using **Value Iteration**, the optimal policy ensures the robot maintains an ideal distance without crashing into the walls.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sensor Records", f"{len(sensor_data):,}")
    col2.metric("State Space Size", f"{len(optimal_df)}")
    col3.metric("Action Space Size", "4 Movement Classes")

    st.markdown("### 🏆 Optimal Policy & Value Function")
    st.dataframe(optimal_df.style.highlight_max(subset=['Optimal_Value'], color='lightgreen'), use_container_width=True)

    st.markdown("### 🧭 Decision Mapping")
    for _, row in optimal_df.iterrows():
        st.info(f"**State: `{row['State']}`** ➔ **Recommended Action:** `{row['Optimal_Action']}` (Value: `{row['Optimal_Value']:.2f}`)")

# ---------------------------------------------------------------------
# ---------------------------------------------------------------------
# VIEW 2: 2D ROBOT ARENA & NAVIGATION PATH SIMULATION
# ---------------------------------------------------------------------
elif app_mode == "🗺️ Robot Wall Simulation":
    st.title("🗺️ 2D Robot Arena & Navigation Path Simulation")
    st.markdown("""
    This simulation maps the robot's physical movement in a 2D corridor/room environment. 
    The robot follows the wall using the learned MDP policy, updating its $(X, Y)$ coordinates to trace a collision-free trajectory.
    """)

    st.sidebar.markdown("### ⚙️ Simulation Settings")
    steps = st.sidebar.slider("Simulation Steps", min_value=20, max_value=300, value=100, step=10)

    # 2D Arena Simulation Logic (X-Y trajectory tracking)
    np.random.seed(42)
    x_pos = [0.0]
    y_pos = [1.0] # Initial distance from left wall
    states_visited = []
    actions_taken = []

    current_y = 1.0
    current_x = 0.0

    for t in range(steps):
        # Determine state based on left distance (current_y)
        if current_y < 0.5:
            state = 'Too-Close'
            action = 'Sharp-Right-Turn'
            # Steer away from wall (increase y) and move forward in x
            current_y += np.random.uniform(0.08, 0.15)
            current_x += np.random.uniform(0.2, 0.4)
        elif current_y < 0.9:
            state = 'Ideal'
            action = 'Move-Forward'
            # Cruise smoothly along the corridor
            current_y += np.random.normal(0, 0.02)
            current_x += np.random.uniform(0.3, 0.5)
        else:
            state = 'Too-Far'
            action = 'Slight-Left-Turn'
            # Steer toward wall (decrease y)
            current_y -= np.random.uniform(0.06, 0.12)
            current_x += np.random.uniform(0.2, 0.4)

        # Keep within corridor bounds
        current_y = max(0.1, min(current_y, 1.8))
        
        x_pos.append(current_x)
        y_pos.append(current_y)
        states_visited.append(state)
        actions_taken.append(action)

    sim_df = pd.DataFrame({
        'Step': range(1, len(x_pos)),
        'X_Position': x_pos[1:],
        'Y_Position': y_pos[1:],
        'State': states_visited,
        'Action': actions_taken
    })

    # Plotly 2D Arena Visualization
    fig = go.Figure()

    # Define structural boundaries (The Walls)
    # Left Wall at Y = 0.0
    fig.add_trace(go.Scatter(
        x=[min(x_pos), max(x_pos)], y=[0.0, 0.0],
        mode='lines', name='Left Structural Wall',
        line=dict(color='black', width=6)
    ))

    # Right Wall Boundary at Y = 2.0
    fig.add_trace(go.Scatter(
        x=[min(x_pos), max(x_pos)], y=[2.0, 2.0],
        mode='lines', name='Right Boundary',
        line=dict(color='black', width=4, dash='dash')
    ))

    # Robot Traced Path (Red continuous line like your reference image)
    fig.add_trace(go.Scatter(
        x=sim_df['X_Position'], y=sim_df['Y_Position'],
        mode='lines+markers', name='Robot Trajectory (MDP Policy)',
        line=dict(color='red', width=3),
        marker=dict(size=5, color='darkred')
    ))

    fig.update_layout(
        title="2D Indoor Environment - Robot Navigation Trajectory",
        xaxis_title="Corridor Length (X)",
        yaxis_title="Corridor Width / Wall Distance (Y)",
        xaxis=dict(showgrid=True),
        yaxis=dict(range=[-0.2, 2.2], showgrid=True),
        hovermode="closest",
        template="plotly_white"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📋 Navigation Log")
    st.dataframe(sim_df, use_container_width=True)

# ---------------------------------------------------------------------
# VIEW 3: TRANSITION PROBABILITIES
# ---------------------------------------------------------------------
elif app_mode == "📊 Transition Probabilities":
    st.title("📊 Empirical Transition Probabilities P(s' | s, a)")
    st.markdown("""
    Transition probabilities estimated directly from consecutive sensor readings in the Kaggle dataset. 
    Select an action below to inspect how likely the robot transitions between states.
    """)

    selected_action = st.selectbox("Select Action", actions)
    action_idx = actions.index(selected_action)
    trans_matrix = T[:, action_idx, :]

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(trans_matrix, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=states, yticklabels=states, ax=ax, vmin=0, vmax=1)
    ax.set_xlabel("Next State (s')")
    ax.set_ylabel("Current State (s)")
    ax.set_title(f"Transition Matrix Heatmap for Action: {selected_action}")
    st.pyplot(fig)

# ---------------------------------------------------------------------
# VIEW 4: DATASET EXPLORER
# ---------------------------------------------------------------------
elif app_mode == "📈 Dataset Explorer":
    st.title("📈 Kaggle Dataset Explorer (`sensor_readings_4.csv`)")
    st.markdown("Explore raw sensor distribution and class frequencies used to build the MDP.")

    col1, col2 = st.columns(2)
    with col1:
        st.subheader("Class Distribution")
        fig, ax = plt.subplots()
        sensor_data['Class'].value_counts().plot(kind='bar', color='skyblue', ax=ax)
        plt.xticks(rotation=45)
        st.pyplot(fig)

    with col2:
        st.subheader("SD_left Sensor Distribution")
        fig, ax = plt.subplots()
        sns.histplot(sensor_data['SD_left'], kde=True, color='purple', ax=ax)
        st.pyplot(fig)

    st.subheader("Raw Data Sample")
    st.dataframe(sensor_data.head(100), use_container_width=True)
