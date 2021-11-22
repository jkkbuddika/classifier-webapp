import matplotlib.pyplot as plt
import streamlit as st

class Plotter:
    def __init__(self, x_data, y_data, title, x_label, y_label, color_array, color_map):
        self.x_data = x_data
        self.y_data = y_data
        self.title = title
        self.x_label = x_label
        self.y_label = y_label
        self.color_array = color_array
        self.color_map = color_map

    def plot_scatter(self):
        fig, ax = plt.subplots(figsize=(6,6))
        ax.scatter(self.x_data, self.y_data, c=self.color_array, cmap=self.color_map)
        ax.set_title(self.title)
        ax.set_xlabel(self.x_label)
        ax.set_ylabel(self.y_label)
        st.pyplot(fig)
