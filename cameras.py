import os

def get_available_cameras():
    camera_devices = []
    for device_name in os.listdir('/dev'):
        if device_name.startswith('video'):
            camera_devices.append(device_name)
    return camera_devices

if __name__ == "__main__":
    available_cameras = get_available_cameras()
    if available_cameras:
        print("Available cameras:")
        for device_name in available_cameras:
            print(f"/dev/{device_name}")
    else:
        print("No cameras detected.")
