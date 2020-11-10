
import os, sys, getopt, gzip, json

def parse_data(in_dir, out_dir):

    # Finding files
    if in_dir.endswith('train'):
        dial_dir = os.path.join(in_dir, 'dialogues_train.txt')
        emo_dir = os.path.join(in_dir, 'dialogues_emotion_train.txt')
        act_dir = os.path.join(in_dir, 'dialogues_act_train.txt')
        out_datset_dir = os.path.join(out_dir, 'dd_datset_training.txt')
    elif in_dir.endswith('validation'):
        dial_dir = os.path.join(in_dir, 'dialogues_validation.txt')
        emo_dir = os.path.join(in_dir, 'dialogues_emotion_validation.txt')
        act_dir = os.path.join(in_dir, 'dialogues_act_validation.txt')
        out_datset_dir = os.path.join(out_dir, 'dd_datset_validation.txt')
    elif in_dir.endswith('test'):
        dial_dir = os.path.join(in_dir, 'dialogues_test.txt')
        emo_dir = os.path.join(in_dir, 'dialogues_emotion_test.txt')
        act_dir = os.path.join(in_dir, 'dialogues_act_test.txt')
        out_datset_dir = os.path.join(out_dir, 'dd_datset_test.txt')
    else:
        print("Cannot find directory")
        sys.exit()

    # Open files
    in_dial = open(dial_dir, 'r')
    in_emo = open(emo_dir, 'r')
    in_act = open(act_dir, 'r')

    out_datset = open(out_datset_dir, mode='a')

    pad = "PAD"
    act_label = {1:'I',2:'Q',3:'D',4:'C'}

    for line_count, (line_dial, line_emo, line_act) in enumerate(zip(in_dial, in_emo, in_act)):

        seqs = line_dial.split('__eou__') # utterrance level
        seqs = seqs[:-1]
        emos = line_emo.split(' ')
        emos = emos[:-1]
        acts = line_act.split(' ')
        acts = acts[:-1]

        seq_len = len(seqs)
        emo_len = len(emos)
        act_len = len(acts)

        speakerids = [pad, pad, pad, pad]
        utts = [pad, pad, pad, pad]
        labels = [pad, pad, pad, pad]

        if seq_len != emo_len or seq_len != act_len:
            print("Different turns btw dialogue & emotion & acttion! ", line_count+1, seq_len, emo_len, act_len)
            sys.exit()

        for turn_count, (seq, emo, act) in enumerate(zip(seqs, emos, acts)):

            # Get rid of the blanks at the start & end of each turns
            if seq[0] == ' ':
                seq = seq[1:]
            if seq[-1] == ' ':
                seq = seq[:-1]

            speakerids.append('{}'.format(1) if (turn_count%2==0) else '{}'.format(2))
            utts.append(seq)
            labels.append(act_label[int(act)])

            line = json.dumps({
                'speakerid': speakerids[-4:],
                'main_topics': [pad, pad, pad, pad],
                'pos': [pad, pad, pad, pad],
                'utt': utts[-4:],
                'label': labels[-4:]
            })

            out_datset.write('{}\n'.format(line))

    in_dial.close()
    in_emo.close()
    in_act.close()
    out_datset.close()



# ==================== Main Function ==================== #
def main(argv):

    in_dir = ''
    out_dir = ''

    try:
        opts, args = getopt.getopt(argv,"h:i:o:",["in_dir=","out_dir="])
    except getopt.GetoptError:
        print("python3 parser.py -i <in_dir> -o <out_dir>")
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print("python3 parser.py -i <in_dir> -o <out_dir>")
            sys.exit()
        elif opt in ("-i", "--in_dir"):
            in_dir = arg
        elif opt in ("-o", "--out_dir"):
            out_dir = arg

    print("Input directory : ", in_dir)
    print("Ouptut directory: ", out_dir)

    parse_data(in_dir, out_dir)

if __name__ == '__main__':
    main(sys.argv[1:])
