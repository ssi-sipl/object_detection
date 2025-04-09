import degirum as dg

inference_manager = dg.connect(
    inference_host_address = dg.LOCAL 
)

supported_types = inference_manager.supported_device_types()
print(f"Supported device types: {supported_types}")

