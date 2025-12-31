import jetson.inference
import jetson.utils

# Initialize PeopleNet model
net = jetson.inference.detectNet(
    argv=[
        '--model=peoplenet',
        '--threshold=0.5',
        '--input-blob=input_0',
        '--output-cvg=scores',
        '--output-bbox=boxes',
        '--batch-size=1',
        '--memory-type=device'
    ]
)

# Camera + Display
camera = jetson.utils.videoSource("csi://0")
display = jetson.utils.videoOutput("display://0")

while display.IsStreaming():
    img = camera.Capture()

    detections = net.Detect(img)

    for det in detections:
        print(
            f"ID:{det.ClassID}  Conf:{det.Confidence:.2f}  "
            f"BBox:({det.Left:.0f},{det.Top:.0f})-({det.Right:.0f},{det.Bottom:.0f})"
        )

    # show frame
    display.Render(img)
    display.SetStatus(f"PeopleNet | FPS: {net.GetNetworkFPS():.1f}")

print("Stream ended — exiting.")
