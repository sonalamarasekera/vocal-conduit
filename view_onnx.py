# Run `pip install netron portpicker` initially

import netron
import portpicker
from google.colab import output

# 1. Choose a port
port = portpicker.pick_unused_port()

# 2. Start the Netron server for your file
# Replace 'your_model.onnx' with the actual path to your file
model_path = "/content/sherpa-onnx-zh-wenet-wenetspeech/model.int8.onnx"
netron.start(model_path, port, browse=False)

# 3. Serve the port as an iframe in the notebook
output.serve_kernel_port_as_iframe(port, height='800')