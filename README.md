# Trackers

## Installion
- Installation
  ```shell
  pip install -v -e .
  ```
## TODO
- [ ] Two line Counter.

## Support
- Track
  * Sort
  * Deepsort
    + 128
    + 512
  * ByteTrack
- Count the person go in or out(one line for now).
- Follow person detect.

## Usage
- Track
  ```python
  # type support `sort`, `deepsort`, `bytetrack`
  tracker = ObjectTracker(type=type)
  # bytetrack need lower conf_thresh.
  conf_thresh = 0.2 if type == "bytetrack" else 0.4
  for img in video:
    # torch.Tensor(N, 6)
    img_raw = img.copy()
    preds = detector.detect(
        img, conf_threshold=conf_thresh
    )
    box = preds[:, :4].cpu()
    conf = preds[:, 4].cpu()
    cls = preds[:, 5].cpu()
    tracks = tracker.update(
        bboxes=box,
        scores=conf,
        cls=cls,
        ori_img=img_raw,  # deepsort need original image to cut and get embedding.
    )

    for i, track in enumerate(tracks):
        box = [int(b) for b in track[:4]]
        id = track[4]
  ```
- Count the person go in or out.
- Follow person detect.

More details see dir `demo`.

## Reference
- ByteTrack
- Sort
- DeepSort
