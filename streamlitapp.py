import pandas as pd
import numpy as np
import streamlit as st
import io
import matplotlib.pyplot as plt
import seaborn as sns

# Initialize session state
if 'data_history' not in st.session_state:
    st.session_state.data_history = []
    st.session_state.current_step = -1

def save_state(data):
    st.session_state.current_step += 1
    st.session_state.data_history = st.session_state.data_history[:st.session_state.current_step]
    st.session_state.data_history.append(data.copy())

def undo():
    if st.session_state.current_step > 0:
        st.session_state.current_step -= 1
    return st.session_state.data_history[st.session_state.current_step]

def redo():
    if st.session_state.current_step < len(st.session_state.data_history) - 1:
        st.session_state.current_step += 1
    return st.session_state.data_history[st.session_state.current_step]

# Set up Streamlit
st.title("Data Analysis Web App")

# Upload file
uploaded_file = st.file_uploader("Upload a CSV or Excel file", type=['csv', 'xlsx'])

if uploaded_file is not None:
    # Read the file
    if uploaded_file.name.endswith('.csv'):
        data = pd.read_csv(uploaded_file)
    else:
        data = pd.read_excel(uploaded_file)

    if st.session_state.current_step == -1:
        save_state(data)

    # Display data summary
    st.subheader("Uploaded Dataset's Summary")
    st.markdown("---")
    st.write(data)
    
    st.subheader("Data Stats")
    st.markdown("---")
    st.write(data.describe())
    
    st.subheader("Data Info")
    st.markdown("---")
    info_buffer = io.StringIO()
    data.info(buf=info_buffer)
    st.text(info_buffer.getvalue())
    
    st.subheader("Shape")
    st.markdown("---")
    st.write(data.shape)
    
    st.subheader("Missing Values")
    st.markdown("---")
    st.write(data.isnull().sum())
    
    st.subheader("Drop Missing Values")
    st.markdown("---")

    if st.button('Drop Rows with Missing Values'):
        data.dropna(inplace=True)
        save_state(data)
        st.write('After dropping rows with any missing values:')
        st.write(data)

    if st.button('Drop Columns with All Missing Values'):
        data.dropna(axis=1, how='all', inplace=True)
        save_state(data)
        st.write('After dropping columns with all missing values:')
        st.write(data)

    if st.button('Drop Columns with Any Missing Values'):
        data.dropna(axis=1, how='any', inplace=True)
        save_state(data)
        st.write('After dropping columns with any missing values:')
        st.write(data)
    
    st.subheader("Filling Missing Values")
    st.markdown("---")

    if st.button('Fill with Mean'):
        data.fillna(data.mean(), inplace=True)
        save_state(data)
        st.write('After filling missing values with mean:')
        st.write(data)

    if st.button('Fill with Median'):
        data.fillna(data.median(), inplace=True)
        save_state(data)
        st.write('After filling missing values with median:')
        st.write(data)

    if st.button('Fill with Mode'):
        mode_values = data.mode().iloc[0]
        data.fillna(mode_values, inplace=True)
        save_state(data)
        st.write('After filling missing values with mode:')
        st.write(data)

    if st.button('Forward Fill (ffill)'):
        data.fillna(method='ffill', inplace=True)
        save_state(data)
        st.write('After forward filling (ffill):')
        st.write(data)

    if st.button('Backward Fill (bfill)'):
        data.fillna(method='bfill', inplace=True)
        save_state(data)
        st.write('After backward filling (bfill):')
        st.write(data)

    if st.button('Interpolate'):
        # Handle interpolation for non-numeric columns
        data = data.apply(pd.to_numeric, errors='ignore')
        data.interpolate(method='linear', inplace=True)
        save_state(data)
        st.write('After interpolating missing values:')
        st.write(data)

    # Undo and Redo buttons
    st.subheader("Undo/Redo")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("Undo"):
            if st.session_state.current_step > 0:
                data = undo()
                st.write('After undo:')
                st.write(data)

    with col2:
        if st.button("Redo"):
            if st.session_state.current_step < len(st.session_state.data_history) - 1:
                data = redo()
                st.write('After redo:')
                st.write(data)

    # Plotting options
    st.subheader("Plotting")
    st.markdown("---")

    plot_type = st.selectbox("Select Plot Type", ["Scatter Plot", "Line Plot", "Bar Plot", "Histogram", "Box Plot", "Heatmap"])

    if plot_type == "Scatter Plot":
        x_col = st.selectbox("Select X Column", data.columns)
        y_col = st.selectbox("Select Y Column", data.columns)
        if st.button("Generate Scatter Plot"):
            plt.figure(figsize=(10, 6))
            sns.scatterplot(x=data[x_col], y=data[y_col])
            plt.title(f'Scatter Plot between {x_col} and {y_col}')
            st.pyplot(plt)

    elif plot_type == "Line Plot":
        x_col = st.selectbox("Select X Column", data.columns)
        y_col = st.selectbox("Select Y Column", data.columns)
        if st.button("Generate Line Plot"):
            plt.figure(figsize=(10, 6))
            sns.lineplot(x=data[x_col], y=data[y_col])
            plt.title(f'Line Plot between {x_col} and {y_col}')
            st.pyplot(plt)

    elif plot_type == "Bar Plot":
        x_col = st.selectbox("Select X Column", data.columns)
        y_col = st.selectbox("Select Y Column", data.columns)
        if st.button("Generate Bar Plot"):
            plt.figure(figsize=(10, 6))
            sns.barplot(x=data[x_col], y=data[y_col])
            plt.title(f'Bar Plot between {x_col} and {y_col}')
            st.pyplot(plt)

    elif plot_type == "Histogram":
        col = st.selectbox("Select Column", data.columns)
        bins = st.slider("Select Number of Bins", min_value=5, max_value=50, value=20)
        if st.button("Generate Histogram"):
            plt.figure(figsize=(10, 6))
            sns.histplot(data[col], bins=bins)
            plt.title(f'Histogram of {col}')
            st.pyplot(plt)

    elif plot_type == "Box Plot":
        x_col = st.selectbox("Select X Column", data.columns)
        y_col = st.selectbox("Select Y Column", data.columns)
        if st.button("Generate Box Plot"):
            plt.figure(figsize=(10, 6))
            sns.boxplot(x=data[x_col], y=data[y_col])
            plt.title(f'Box Plot of {y_col} by {x_col}')
            st.pyplot(plt)

    elif plot_type == "Heatmap":
        if st.button("Generate Heatmap"):
            plt.figure(figsize=(10, 6))
            sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
            plt.title('Heatmap of Correlation Matrix')
            st.pyplot(plt)

    st.markdown("---")

    # Display updated data summary
    st.subheader("Updated Dataset's Summary")
    st.markdown("---")
    st.write(data)
    
    st.subheader("Updated Data Stats")
    st.markdown("---")
    st.write(data.describe())
    
    st.subheader("Updated Data Info")
    st.markdown("---")
    info_buffer_filled = io.StringIO()
    data.info(buf=info_buffer_filled)
    st.text(info_buffer_filled.getvalue())
    
    st.subheader("Updated Shape")
    st.markdown("---")
    st.write(data.shape)
