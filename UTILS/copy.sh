count_0=0
count_1=0
count_2=0
count_3=0
count_4=0
count_5=0
count_6=0
count_7=0

total=0

for f in ../Datasets/Facial\ Emotion\ Recognition/AffectNet/val_set/annotations/*; do
    if [[ $f == *"exp"* ]]; then
        image_id=$(echo $f | tr -d -c 0-9)
        emotion_id=$(tail -1 $f)
        if [[ $emotion_id == "0" ]]; then
            emotion="Neutral"
            count_0=$((count_0+1))
            number=$(printf %07d $count_0)
            destination="../Datasets/Facial\ Emotion\ Recognition/AffectNet/Annotated/images/"$emotion"/"$number".jpg"
            source="../Datasets/Facial\ Emotion\ Recognition/AffectNet/val_set/images/"$image_id".jpg"
            cp $source $destination
        fi
        if [[ $emotion_id == "1" ]]; then
            emotion="Happy"
            count_1=$((count_1+1))
            number=$(printf %07d $count_1)
            destination="../Datasets/Facial\ Emotion\ Recognition/AffectNet/Annotated/images/"$emotion"/"$number".jpg"
            source="../Datasets/Facial\ Emotion\ Recognition/AffectNet/val_set/images/"$image_id".jpg"
            cp $source $destination
        fi
        if [[ $emotion_id == "2" ]]; then
            emotion="Sad"
            count_2=$((count_2+1))
            number=$(printf %07d $count_2)
            destination="../Datasets/Facial\ Emotion\ Recognition/AffectNet/Annotated/images/"$emotion"/"$number".jpg"
            source="../Datasets/Facial\ Emotion\ Recognition/AffectNet/val_set/images/"$image_id".jpg"
            cp $source $destination
        fi
        if [[ $emotion_id == "3" ]]; then
            emotion="Surprise"
            count_3=$((count_3+1))
            number=$(printf %07d $count_3)
            destination="../Datasets/Facial\ Emotion\ Recognition/AffectNet/Annotated/images/"$emotion"/"$number".jpg"
            source="../Datasets/Facial\ Emotion\ Recognition/AffectNet/val_set/images/"$image_id".jpg"
            cp $source $destination
        fi
        if [[ $emotion_id == "4" ]]; then
            emotion="Fear"
            count_4=$((count_4+1))
            number=$(printf %07d $count_4)
            destination="../Datasets/Facial\ Emotion\ Recognition/AffectNet/Annotated/images/"$emotion"/"$number".jpg"
            source="../Datasets/Facial\ Emotion\ Recognition/AffectNet/val_set/images/"$image_id".jpg"
            cp $source $destination
        fi
        if [[ $emotion_id == "5" ]]; then
            emotion="Disgust"
            count_5=$((count_5+1))
            number=$(printf %07d $count_5)
            destination="../Datasets/Facial\ Emotion\ Recognition/AffectNet/Annotated/images/"$emotion"/"$number".jpg"
            source="../Datasets/Facial\ Emotion\ Recognition/AffectNet/val_set/images/"$image_id".jpg"
            cp $source $destination

        fi
        if [[ $emotion_id == "6" ]]; then
            emotion="Anger"
            count_6=$((count_6+1))
            number=$(printf %07d $count_6)
            destination="../Datasets/Facial\ Emotion\ Recognition/AffectNet/Annotated/images/"$emotion"/"$number".jpg"
            source="../Datasets/Facial\ Emotion\ Recognition/AffectNet/val_set/images/"$image_id".jpg"
            cp $source $destination
        fi
        if [[ $emotion_id == "7" ]]; then
            emotion="Contempt"
            count_7=$((count_7+1))
            number=$(printf %07d $count_7)
            destination="../Datasets/Facial\ Emotion\ Recognition/AffectNet/Annotated/images/"$emotion"/"$number".jpg"
            source="../Datasets/Facial\ Emotion\ Recognition/AffectNet/val_set/images/"$image_id".jpg"
            cp $source $destination
        fi
        total=$((total+1))
        if [[ $(expr $total % 1000) == "0" ]]; then
            echo $total" Done"
        fi
    fi
done