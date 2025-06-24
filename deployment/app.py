import gradio as gr
from inference import model_fn, predict_fn
import os

# Emoji mappings
EMOTION_EMOJI = {
    'anger': '😡',
    'disgust': '🤢',
    'fear': '😨',
    'joy': '😄',
    'neutral': '😐',
    'sadness': '😢',
    'surprise': '😲'
}
SENTIMENT_EMOJI = {
    'negative': '😡',
    'neutral': '😐',
    'positive': '😄'
}

# Load model once
model_dict = model_fn('.')

# Utility to compute top emotion/sentiment across utterances
def compute_overall(utterances):
    # aggregate confidences
    emo_scores = {}
    senti_scores = {}
    for utt in utterances:
        for emo in utt.get('emotions', []):
            emo_scores.setdefault(emo['label'], []).append(emo['confidence'])
        for sent in utt.get('sentiments', []):
            senti_scores.setdefault(sent['label'], []).append(sent['confidence'])
    # compute average
    def avg(scores): return sum(scores)/len(scores) if scores else 0
    emo_avg = [(label, avg(scores)) for label, scores in emo_scores.items()]
    senti_avg = [(label, avg(scores)) for label, scores in senti_scores.items()]
    if not emo_avg or not senti_avg:
        return None
    # pick top
    top_emo_label, top_emo_score = max(emo_avg, key=lambda x: x[1])
    top_sent_label, top_sent_score = max(senti_avg, key=lambda x: x[1])
    return {
        'emotion_label': top_emo_label,
        'emotion_score': top_emo_score,
        'emotion_emoji': EMOTION_EMOJI.get(top_emo_label, ''),
        'sentiment_label': top_sent_label,
        'sentiment_score': top_sent_score,
        'sentiment_emoji': SENTIMENT_EMOJI.get(top_sent_label, '')
    }


def predict(video):
    temp_path = "temp.mp4"
    try:
        # save upload
        if isinstance(video, str) and os.path.exists(video):
            os.rename(video, temp_path)
        else:
            with open(temp_path, 'wb') as f:
                f.write(video.read())
        # run inference
        result = predict_fn({'video_path': temp_path}, model_dict)
        utterances = result.get('utterances', [])
        # overall analysis
        overall = compute_overall(utterances) or {}
        overall_html = f"""
        <div class='overall-card'>
            <div>
                <strong>Primary Emotion</strong><br>
                <span class='emoji'>{overall.get('emotion_emoji')}</span> {overall.get('emotion_label', '')}<br>
                <small>{overall.get('emotion_score', 0)*100:.1f}%</small>
            </div>
            <div>
                <strong>Primary Sentiment</strong><br>
                <span class='emoji'>{overall.get('sentiment_emoji')}</span> {overall.get('sentiment_label', '')}<br>
                <small>{overall.get('sentiment_score', 0)*100:.1f}%</small>
            </div>
        </div>
        """
        # utterance cards
        utt_html = ''
        for utt in utterances:
            utt_html += f"""
            <div class='utt-card'>
                <div class='time'>{utt['start_time']:.1f}s - {utt['end_time']:.1f}s</div>
                <div class='text'>{utt['text']}</div>
                <div class='bar-group'><span>Emotions</span>"""
            for emo in utt.get('emotions', []):
                pct = emo['confidence'] * 100
                utt_html += f"<div class='bar'><div class='fill emo' style='width:{pct:.0f}%'></div><span>{emo['label']}: {pct:.0f}%</span></div>"
            utt_html += "</div><div class='bar-group'><span>Sentiments</span>"
            for sent in utt.get('sentiments', []):
                pct = sent['confidence'] * 100
                utt_html += f"<div class='bar'><div class='fill senti' style='width:{pct:.0f}%'></div><span>{sent['label']}: {pct:.0f}%</span></div>"
            utt_html += "</div></div>"
        return overall_html + utt_html
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

# Dark-mode CSS
gr_css = """
body { background: #121212; color: #e0e0e0; }
.overall-card { display: flex; justify-content: space-around; background: #1f1f1f; padding: 16px; border-radius: 10px; margin-bottom: 16px; }
.overall-card .emoji { font-size: 2rem; display: block; margin: 4px 0; }
.utt-card { background: #1f1f1f; padding: 12px; margin-bottom: 12px; border-radius: 10px; }
.time { font-size: 0.85rem; color: #888; margin-bottom: 4px; }
.text { font-size: 0.95rem; margin-bottom: 8px; }
.bar-group { margin-bottom: 8px; }
.bar-group span { font-weight: 600; display: block; margin-bottom: 4px; }
.bar { position: relative; background: #2a2a2a; height: 14px; border-radius: 7px; margin-bottom: 4px; overflow: hidden; }
.fill { height: 100%; }
.fill.emo { background: #e67e22; }
.fill.senti { background: #27ae60; }
.bar span { position: absolute; top: 0; left: 6px; font-size: 0.75rem; line-height: 14px; }
"""

# Build Gradio App
with gr.Blocks(css=gr_css) as demo:
    gr.Markdown('# 🎬 Video Sentiment & Emotion Analyzer')
    with gr.Row():
        with gr.Column(scale=1):
            vid_in = gr.Video(label='Upload a video')
            btn = gr.Button('Analyze', variant='primary')
        with gr.Column(scale=2):
            out = gr.HTML()
    btn.click(predict, inputs=vid_in, outputs=out)

if __name__ == '__main__':
    demo.launch()
