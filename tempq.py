import plotly.graph_objects as go
from keras.models import Sequential
from keras.layers import Dense

# Define the neural network architecture
input_dim = 7 + 9
model = Sequential()
model.add(Dense(784, activation='relu', name='input_layer'))
model.add(Dense(100, activation='relu', name='hidden_layer1'))
model.add(Dense(10, activation='softmax', name='output_layer'))

# Create a plotly figure
fig = go.Figure()

# Add nodes for each layer
for layer in model.layers:
    if isinstance(layer, Dense):
        # Add nodes for each neuron in the layer
        for i in range(layer.units):
            neuron_id = f"{layer.name}_{i}"
            fig.add_trace(go.Scatter(x=[neuron_id], y=[i], mode='markers', marker=dict(size=10)))

# Connect nodes with edges
for i in range(len(model.layers) - 1):
    layer1 = model.layers[i]
    layer2 = model.layers[i + 1]

    if isinstance(layer1, Dense) and isinstance(layer2, Dense):
        for i in range(layer1.units):
            for j in range(layer2.units):
                neuron_id1 = f"{layer1.name}_{i}"
                neuron_id2 = f"{layer2.name}_{j}"
                fig.add_trace(go.Scatter(x=[neuron_id1, neuron_id2], y=[i, j], mode='lines'))

# Customize layout
fig.update_layout(title_text="Neural Network Architecture", showlegend=False)

# Show the figure
fig.show()
