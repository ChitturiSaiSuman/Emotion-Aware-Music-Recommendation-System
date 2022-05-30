import streamlit, UTILS, PIL, numpy
import streamlit.components.v1 as components

def populate_tracks(tracks: list) -> None:
    tag = "<h4 style='text-align: center;'>Top Tracks</h4>"
    streamlit.markdown(tag, unsafe_allow_html = True)
    with streamlit.container():
        col1, col2, col3 = streamlit.columns([3, 3, 3])
        for i, track in enumerate(tracks):
            if i % 3 == 0:
                with col1:
                    components.html(frame.format(track), height = 400)
            elif i % 3 == 1:
                with col2:
                    components.html(frame.format(track), height = 400)
            else:
                with col3:
                    components.html(frame.format(track), height = 400)

def populate_emotion(emotion: str) -> None:
    left_color = UTILS.constants['colors']['left']
    left_span = '<span style = "color: {}">Identified Emotion: </span>'
    left_span = left_span.format(left_color)

    emotion_color = UTILS.constants['colors'][emotion]
    right_span = '<span style = "color: {}">{}</span>'
    right_span = right_span.format(emotion_color, emotion.capitalize())
    
    tag = "<h2 style='text-align: center;'>{}</h2>"
    tag = tag.format(left_span + right_span)
    streamlit.markdown(tag, unsafe_allow_html=True)

if __name__ == '__main__':
    tracks = UTILS.tracks
    frame = UTILS.constants['frame']
    picture = streamlit.camera_input('')
    if picture is not None:
        picture = PIL.Image.open(picture)
        pixels = numpy.array(picture)
        emotion = UTILS.detect_emotion(pixels)
        if emotion:
            tracks = UTILS.get_top_k(emotion)
            populate_emotion(emotion)
            populate_tracks(tracks)
        else:
            tag = '<h3 style="text-align: center;">No face detected</h3>'
            streamlit.markdown(tag, unsafe_allow_html = True)