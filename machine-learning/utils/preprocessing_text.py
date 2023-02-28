import string

def removePunctuation(lines):
    # Remove punctuation
    lines[:,0] = [s.translate(str.maketrans('', '', string.punctuation)) for s in lines[:,0]]
    lines[:,1] = [s.translate(str.maketrans('', '', string.punctuation)) for s in lines[:,1]]
    return lines

def toLowercase(lines):
    for i in range(len(lines)):
        lines[i,0] = lines[i,0].lower()   
        lines[i,1] = lines[i,1].lower()
    return lines