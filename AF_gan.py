import os


def read_file(filename):
    data = {'time': [], 'accel': [], 'intaccel': [], 'sg1': [], 'sg2': [], 'sg3': []}
    with open(filename, 'r') as tempFile:
        tempFile.readline()
        tempFile.readline()
        data_line = tempFile.readline()
        while data_line:
            split = data_line.split()
            data['time'].append(float(split[0]))
            data['accel'].append(float(split[1]))
            data['intaccel'].append(float(split[2]))
            data['sg1'].append(float(split[3]))
            data['sg2'].append(float(split[4]))
            data['sg3'].append(float(split[5]))
            data_line = tempFile.readline()
    return data

if __name__ == '__main__':
    print(os.getcwd())

    temp_data = read_file('signal_data/T05.txt')
    for key in temp_data:
        print('First four: {}, last: {}'.format(temp_data[key][0:4], temp_data[key][-1]))