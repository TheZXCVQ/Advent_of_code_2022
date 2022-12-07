from aocd.models import Puzzle
from queue import PriorityQueue
from collections import deque
import string


def puzzle_2022_1_1(input=None):
    maxi = 0
    buff = 0
    for line in input.split('\n'):
        if line == "":
            if buff > maxi:
                maxi = buff
            buff = 0
        else:
            buff += int(line)
    if buff > maxi:
        maxi = buff
    return maxi


def puzzle_2022_1_2(input=None):
    q = PriorityQueue(maxsize=4)
    ans = 0
    buff = 0
    for line in input.split('\n'):
        if line == "":
            q.put((buff, buff))
            if (q.full()):
                q.get()
            buff = 0
        else:
            buff += int(line)
    for i in range(3):
        ans += q.get()[0]
    return ans


def puzzle_2022_2_1(input=None):
    score = 0
    for line in input.split('\n'):
        opponent = ord(line[0]) - ord('A')
        me = ord(line[2]) - ord('X')
        score += 1 + me + ((((me - opponent) % 3) + 1) % 3) * 3
    return score


def puzzle_2022_2_2(input=None):
    score = 0
    for line in input.split('\n'):
        opponent = ord(line[0]) - ord('A')
        result = ord(line[2]) - ord('X')
        score += 1 + ((opponent + (result + 2) % 3) % 3) + result * 3
    return score


def puzzle_2022_3_1(input=None):
    def evaluate(a):
        num = ord(a)
        return (1 + num - ord('a')) if num >= ord('a') else (27 + num - ord('A'))

    ans = 0
    for line in input.split('\n'):
        length = len(line) // 2
        first_str = line[:length]
        second_str = line[length:]
        first_dic = {}
        second_dic = {}
        for i in range(length):
            a = first_str[i]
            b = second_str[i]
            if (a not in first_dic):
                first_dic[a] = evaluate(a)
            if (b not in second_dic):
                second_dic[b] = evaluate(b)
        difference_dic = dict(first_dic.items() & second_dic.items())
        for i in difference_dic:
            ans += difference_dic[i]
    return ans


def puzzle_2022_3_2(input=None):
    def evaluate(a):
        num = ord(a)
        return (1 + num - ord('a')) if num >= ord('a') else (27 + num - ord('A'))

    ans = 0
    for line in zip(input.split('\n')[::3], input.split('\n')[1::3], input.split('\n')[2::3]):
        dics = [{}, {}, {}]
        for i in range(3):
            for j in line[i]:
                if (j not in dics[i]):
                    dics[i][j] = evaluate(j)

        difference_dic = dict(dics[0].items() & dics[1].items() & dics[2].items())
        for i in difference_dic:
            ans += difference_dic[i]
    return ans


def puzzle_2022_4_1(input=None):
    ans = 0
    for line in input.split('\n'):
        edges_list = []
        for i in line.split(','):
            for j in i.split('-'):
                edges_list.append(int(j))
        if ((edges_list[2] <= edges_list[0] and edges_list[1] <= edges_list[3]) or (
                edges_list[0] <= edges_list[2] and edges_list[3] <= edges_list[1])):
            ans += 1
    return ans


def puzzle_2022_4_2(input=None):
    ans = 0
    for line in input.split('\n'):
        edges_list = []
        for i in line.split(','):
            for j in i.split('-'):
                edges_list.append(int(j))
        if (edges_list[1] > edges_list[3]):
            (edges_list[0], edges_list[1], edges_list[2], edges_list[3]) = (
                edges_list[2], edges_list[3], edges_list[0], edges_list[1])
        if (edges_list[2] <= edges_list[1]):
            ans += 1
    return ans


def puzzle_2022_5_1(input=None):
    move_flag = False
    nr_of_stacks = (len(input.split('\n')[0]) + 1) // 4
    stacks = [deque() for _ in range(nr_of_stacks)]
    ans = ""
    for line in input.split('\n'):
        if (not move_flag):
            if (len(line) == 0):
                move_flag = True
            else:
                for i in range(nr_of_stacks):
                    if (line[1 + 4 * i] != ' '):
                        stacks[i].appendleft(line[1 + 4 * i])
        else:
            command = line.split(" ")
            for i in range(int(command[1])):
                stacks[int(command[5]) - 1].append(stacks[int(command[3]) - 1].pop())
    for st in stacks:
        ans += st.pop()
    return ans


def puzzle_2022_5_2(input=None):
    move_flag = False
    nr_of_stacks = (len(input.split('\n')[0]) + 1) // 4
    stacks = [[] for _ in range(nr_of_stacks)]
    ans = ""
    for line in input.split('\n'):
        if (not move_flag):
            if (len(line) == 0):
                move_flag = True
            else:
                for i in range(nr_of_stacks):
                    if (line[1 + 4 * i] != ' '):
                        stacks[i].insert(0, line[1 + 4 * i])
        else:
            command = line.split(" ")
            stacks[int(command[5]) - 1] += (stacks[int(command[3]) - 1][-int(command[1]):])
            stacks[int(command[3]) - 1] = stacks[int(command[3]) - 1][:-int(command[1])]
    for st in stacks:
        ans += st.pop()
    return ans


def puzzle_2022_6_1(input=None):
    l = 0
    for r in range(0, len(input) - 1):
        if (r - l >= 4):
            return r
        for i in range(l, r + 1):
            if (input[i] == input[r + 1]):
                l = i + 1
                break




def puzzle_2022_6_2(input=None):
    l = 0
    for r in range(0, len(input) - 1):
        if (r - l >= 14):
            return r
        for i in range(l, r + 1):
            if (input[i] == input[r + 1]):
                l = i + 1
                break

class File:

    def __init__(self,name, type, parent, size=None, children=[]):
        self.name=name
        self.type=type
        self.parent=parent
        self.size=size
        self.children=children
    def get_size(self):
        if(self.size==None):
            self.size = sum([_.get_size() for _ in self.children])
        return self.size
    def size_recursion(self, limit):
        if(self.typ=="txt"):
            return 0
        ret_size=0
        if(self.get_size()<=limit):
            ret_size+=self.get_size()
        return ret_size+sum([_.size_recursion(limit) for _ in self.children])
    def get_child(self,child_name):
        for i in self.children:
            if(i.name==child_name):
                return i
        print("no child ",child_name," found")
        return None
    def print(self,offset=0):
        if(offset>=10):
            return
        print(" "*offset,"- ", self.name, " ( ",self.type, " )")
        if(self.type=="dir"):
            for i in self.children:
                i.print(offset+2)



def puzzle_2022_7_1(input=None):
    root=File("root","dir",None)
    wd=None
    k=0
    for line in input.split('\n'):
        com=line.split(' ')
        if(com[1]=="cd"):
            if(wd!=None):
                print(wd.name)
            if(com[2]=="/"):
                wd=root
            elif(com[2]==".."):
                wd=wd.parent
                k-=1
            else:
                k+=1
                if(wd==None):
                    pass
                    #root.print()
                wd=wd.get_child(com[2])
        elif(com[0]!="$"):
            if(com[0]=="dir"):
                wd.children.append(File(com[1],"dir",wd))
            else:
                wd.children.append(File(com[1], "txt",wd , int(com[0])))

    return root.size_recursion(100000)

if __name__ == '__main__':
    year = 2022
    day = 7
    part = 1
    send = False
    puzzle = Puzzle(year=year, day=day, )
    fname = "puzzle_" + str(year) + "_" + str(day) + "_" + str(part)
    answer = globals()[fname](puzzle.input_data)
    print(answer)
    if (send):
        if part == 1:
            puzzle.answer_a = answer
        elif part == 2:
            puzzle.answer_b = answer
