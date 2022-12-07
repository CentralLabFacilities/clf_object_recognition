from vision_msgs.msg import Detection2D, ObjectHypothesisWithPose

def tensorsToVisionMessage(ids, scores, boxes, header, thresh = 0.35):
	detections = []

	for i in range(len(ids)):
		box = boxes[i]
		cls_id = int(ids[i])
		score = scores[i]
		if score < thresh:
			# tensor is sorted by scores
			break

		msg = Detection2D()
		msg.header = header

		x0 = int(box[0])
		y0 = int(box[1])
		x1 = int(box[2])
		y1 = int(box[3])

		msg.bbox.center.x = (x0 + x1) / 2.0
		msg.bbox.center.y = (y0 + y1) / 2.0
		msg.bbox.size_x = x1 - x0
		msg.bbox.size_y = y1 - y0

		hyp = ObjectHypothesisWithPose()
		hyp.id = cls_id
		hyp.score = score

		msg.results.append(hyp)

		detections.append(msg)

	return detections