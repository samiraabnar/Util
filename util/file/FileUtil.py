import re

class FileUtil(object):

    def read_glove_vec(filename):
        vectors = {}
        with open(filename,'r') as filedata:
            for line in filedata.readlines(100000000):
                data = line.split(' ')
                vectors[str(data[0]).lower().strip()] = data[1:]
                print(str(data[0]))
        return vectors;

    def make_glove_words(filename):
        words = []
        with open(filename,'r') as filedata:
            for line in filedata.readlines(100000000):
                data = line.split(' ')
                words.append(str(data[0]).lower().strip())

        FileUtil.list2file(words,'../data/gloveWords.txt')

    def make_glove_vectors(filename):
        vectorz = []
        with open(filename,'r') as filedata:
            for line in filedata.readlines(100000000):
                data = line.split(' ')
                vectorz.append(data[1:])

        FileUtil.list2vectorfile(vectorz,'../data/gloveVectorz.txt')

    def seperate_syn_ants(filename,mainfilename,antfilename):
        words = []
        ants = []
        with open(filename,'r') as filedata:
            for line in filedata:
                parts = line.split()
                words.append(parts[0].strip())
                ants.append(parts[1].strip())

        with open(mainfilename, 'w') as filedata:
                for word in words:
                    filedata.write('%s\n' % word)

        with open(antfilename, 'w') as filedata:
                for word in ants:
                    filedata.write('%s\n' % word)

    def file2list_limited(filename,start,end):
        l = []

        with open(filename,'r') as filedata:
            for line in filedata.readlines()[start:end]:
                l.append(line.lower().strip())

        return l

    def file2list(filename):
        l = []

        with open(filename,'r') as filedata:
            for line in filedata.readlines():
                l.append(line.lower().strip())

        return l

    def list2vectorfile(list,filename):
        with open(filename, 'w') as filedata:
                for item in list:
                        filedata.write("%s" % ','.join(map(str,item)))

    def list2file(list,filename):
        with open(filename, 'w') as filedata:
                for item in list:
                    filedata.write('%s\n' % item)

    def add_line_numbers(filename, samefile):
        with open(filename,'r') as filedata:
            data = filedata.readlines()
        outputfilename = filename
        if samefile == False:
            outputfilename = 'Numbered_'+filename

        with open(outputfilename, 'w') as filedata:
            for(number, line) in enumerate(data):
                filedata.write('%d %s' % (number + 1, line))

    def remove_sentiment_annotations(filename,samefile):
        with open(filename,'r') as filedata:
            data = filedata.readlines()
        outputfilename = filename
        if samefile == False:
            outputfilename = 'AnnotationFree_'+filename

        with open(outputfilename, 'w') as filedata:
            for(number, str) in enumerate(data):
                str = re.sub(r"\s*(\(\d+)", "", str)
                str = re.sub(r"\)", "", str)
                filedata.write('%s\n' % (str.strip()))


    def get_sentence_and_label_from_tree_annotation(filename,samefile=False):
        with open(filename,'r') as filedata:
            data = filedata.readlines()
        outputfilename = filename
        if samefile == False:
            outputfilename = filename.replace(".txt","sentence_and_label.txt")

        with open(outputfilename, 'w') as filedata:
            for(number, annotatedSent) in enumerate(data):
                label_match = re.search(r"\((\d+)\s*\(",annotatedSent)
                label = label_match.group(1)

                annotatedSent = re.sub(r"\s*(\(\d+)", "", annotatedSent)
                annotatedSent = re.sub(r"\)", "", annotatedSent)
                filedata.write('%s %s\n' % (label.strip(),annotatedSent.strip()))


    def file_2_vec(filename,start,end):
        with open(filename,'r') as filedata:
            data = filedata.readlines()[start:end]
        vectors = {}
        for(number, line) in enumerate(data):
            vectors[number] = [ float(item) for item in line.split(',')]

        return vectors

    """def file_2_vec(filename):
        with open(filename,'r') as filedata:
            data = filedata.readlines()
        vectors = {}
        for(number, line) in enumerate(data):
            vectors[number] = [ float(item) for item in line.split(',')]

        return vectors"""

    def file_2_sentList(filename):
        with open(filename,'r') as filedata:
            data = filedata.readlines()
        return data




if __name__ == "__main__":
    #vecz = FileUtil.read_glove_vec('../data/glove.840B.300d.txt')
    FileUtil.make_glove_vectors('../data/glove.840B.300d.txt')
    #print(len(vecz))
    #FileUtil.make_parallel_vectors(vecz)
    #FileUtil.get_parallel_vectors(vecz,FileUtil.file2list('../data/words.txt'),FileUtil.file2list('../data/ants.txt'))
    #FileUtil.seperate_syn_ants('../data/antonyms','../data/words.txt','../data/ants.txt')
