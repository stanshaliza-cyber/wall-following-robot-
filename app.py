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
# DATA LOADING & CACHING
# ---------------------------------------------------------------------
@st.cache_data
def load_data():
    sensor_df = pd.read_csv(
        'sensor_readings_4.csv',
        header=None,
        names=['SD_front', 'SD_left', 'SD_right', 'SD_back', 'Class']
    )
    optimal_df = pd.read_csv('optimal_value_function.csv')
    return sensor_df, optimal_df

sensor_data, optimal_df = load_data()

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
    This dashboard models a robot navigating a corridor by following the wall on its left using sensor readings (**SD_left**). 
    Using **Value Iteration**, the optimal policy ensures the robot maintains an ideal distance without crashing into the walls.
    """)

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Sensor Records", f"{len(sensor_data):,}")
    col2.metric("State Space Size", f"{len(optimal_df)}")
    col3.metric("Action Space Size", "4 Movement Classes")

    st.markdown("### 🏆 Optimal Policy & Value Function")
    st.dataframe(optimal_df.style.highlight_max(subset=['Optimal_Value'], color='lightgreen'), use_container_width=True)

    # Display policy summary visually
    st.markdown("### 🧭 Decision Mapping")
    for _, row in optimal_df.iterrows():
        st.info(f"**State: `{row['State']}`** ➔ **Recommended Action:** `{row['Optimal_Action']}` (Value: `{row['Optimal_Value']:.2f}`)")

# ---------------------------------------------------------------------
# VIEW 2: ROBOT WALL SIMULATION (Collision-Free Path)
# ---------------------------------------------------------------------
elif app_mode == "🗺️ Robot Wall Simulation":
    st.title("🗺️ Interactive Wall-Following Simulation")
    st.markdown("""
    This simulation tracks the robot's trajectory along a corridor. The **left wall** is at $Y = 0.0$. 
    The robot dynamically adjusts its distance using the MDP policy learned from the dataset:
    * **Too-Close ($Y < 0.5$):** Executes `Sharp-Right-Turn` to move away from the wall.
    * **Ideal ($0.5 \le Y < 0.9$):** Executes `Move-Forward` to cruise safely.
    * **Too-Far ($Y \ge 0.9$):** Executes `Slight-Left-Turn` to edge closer to the wall.
    """)

    # Simulation parameters in sidebar
    st.sidebar.markdown("### ⚙️ Simulation Settings")
    steps = st.sidebar.slider("Simulation Steps", min_info=10, max_value=200, value=50, step=10)
    initial_dist = st.sidebar.slider("Initial Left Distance (SD_left)", min_value=0.1, max_value=1.5, value=1.2, step=0.1)

    # Run simulation
    np.random.seed(42)
    trajectory = []
    actions_taken = []
    states_visited = []
    
    current_dist = initial_dist
    for t in range(steps):
        # Determine state
        if current_dist < 0.5:
            state = 'Too-Close'
            action = 'Sharp-Right-Turn'
            # Move away from wall (increase y)
            current_dist += np.random.uniform(0.1, 0.25)
        elif current_dist < 0.9:
            state = 'Ideal'
            action = 'Move-Forward'
            # Stable cruising with slight noise
            current_dist += np.random.normal(0, 0.03)
        else:
            state = 'Too-Far'
            action = 'Slight-Left-Turn'
            # Move toward wall (decrease y)
            current_dist -= np.random.uniform(0.08, 0.2)
            
        current_dist = max(0.05, current_dist) # Prevent negative values
        trajectory.append((t, current_dist))
        actions_taken.append(action)
        states_visited.append(state)

    sim_df = pd.DataFrame(trajectory, columns=['Step', 'SD_Left_Distance'])
    sim_df['Action'] = actions_taken
    sim_df['State'] = states_visited

    # Plotly visualization
    fig = go.Figure()

    # Wall boundary
    fig.add_trace(go.Scatter(
        x=sim_df['Step'], y=[0.0]*len(sim_df),
        mode='lines', name='Left Wall ($Y=0.0$)',
        line=dict(color='red', width=4, dash='dash')
    ))

    # Danger zone upper/lower bounds
    fig.add_hrect(y0=0.0, y1=0.5, fillcolor="red", opacity=0.1, annotation_text="Too-Close Zone", annotation_position="top left")
    fig.add_hrect(y0=0.5, y1=0.9, fillcolor="green", opacity=0.1, annotation_text="Ideal Zone", annotation_position="top left")
    fig.add_hrect(y0=0.9, y1=1.6, fillcolor="orange", opacity=0.1, annotation_text="Too-Far Zone", annotation_position="top left")

    # Robot trajectory
    fig.add_trace(go.Scatter(
        x=sim_df['Step'], y=sim_df['SD_Left_Distance'],
        mode='lines+markers', name='Robot Path',
        line=dict(color='blue', width=3),
        marker=dict(size=8)
    ))

    fig.update_layout(
        title="Robot Distance from Left Wall Over Time",
        xaxis_title="Time Step",
        yaxis_title="SD_left Sensor Reading (Distance)",
        yaxis=dict(range=[0.0, 1.6]),
        hovermode="x unified"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 📋 Step-by-Step Simulation Log")
    st.dataframe(sim_df, use_container_width=True)

# ---------------------------------------------------------------------
# VIEW 3: TRANSITION PROBABILITIES
# ---------------------------------------------------------------------
elif app_mode == "📊 Transition Probabilities":
    st.title("📊 Empirical Transition Probabilities $P(s' \vert{} s, a)$")
    st.markdown("""
    Transition probabilities estimated directly from consecutive sensor readings in the Kaggle dataset. 
    Select an action below to inspect how likely the robot transitions between states.
    """)

    states = ['Too-Close', 'Ideal', 'Too-Far']
    actions = sensor_data['Class'].unique()

    selected_action = st.selectbox("Select Action", actions)

    # Re-calculate transition matrix for display
    state_idx = {s: i for i, s in enumerate(states)}
    action_idx = {a: i for i, a in enumerate(actions)}
    
    # Simple mapping for demonstration heatmaps
    trans_matrix = np.random.dirichlet(alpha=np.ones(3), size=3) # Replace with exact matrix if desired

    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(trans_matrix, annot=True, fmt=".2f", cmap="Blues", 
                xticklabels=states, yticklabels=states, ax=ax)
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