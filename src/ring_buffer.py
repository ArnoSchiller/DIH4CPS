"""
RingBuffer: This class implemtents a data struct equal to a ring buffer.   

It is possible to add elements, if the maximum length is reached the first added 
element would be overwritten. Also it is possible to get elements from the list. 
To prevent the read and the write pointers from blocking each other the index
of the read pointer will be updated automatically. 

Requirements:

Links: 
    - https://towardsdatascience.com/circular-queue-or-ring-buffer-92c7b0193326

@authors:   Arno Schiller (AS)
@email:     schiller@swms.de
@version:   v2.0.0
@license:   see https://github.com/ArnoSchiller/DIH4CPS-PYTESTS

VERSION HISTORY
Version:    (Author) Description:                                   Date:     \n
v0.0.1      (AS) First initialize.                                  31.09.2020\n
v0.0.2      (AS) Added functionality to append and get elements     31.09.2020\n
                from the buffer. Also added function to visualize             \n
                the pointers movement.                                        \n
v2.0.0      (AS) Included into v3, nothing to change.               04.11.2020\n
"""
import threading, time
import numpy as np

class RingBuffer:
    """ 
    This class implemtents a data struct equal to a ring buffer.  
    
    Attributes:
    -----------
    max_len : int
        maximum number of elements the buffer can contain
    head_pos : int
        list index the head is looking at
    tail_pos : int
        list index the tail is looking at
    element_list : list
        buffer list containing included elements
    """
    head_pos = -1 
    tail_pos = 0
    element_list = []
    
    def __init__(self, max_len=256):
        """
        Constructor creates an RingBuffer object.
        Parameters
        ----------
        max_len : int
            number of elements the list can contain  
        """
        self.max_len = max_len

    def append_element(self, element):
        """
        Appends the given element to list. If the maximum number 
        of elements is reached the first added element will be
        overwritten. 
    
        Parameters
        ----------
        element : type
            object to add to the buffer 
        """
        if len(self.element_list) < self.max_len:
            self.element_list.append(element)
        else:
            self.element_list[self.tail_pos] = element 
        self.tail_pos = (self.tail_pos + 1) % self.max_len

    def get_element(self, index):
        """
        Appends the given element to list. If the maximum number 
        of elements is reached the first added element will be
        overwritten. 
    
        Parameters
        ----------
        index : int
            position of the element that should be returned

        Returns
        ----------
        element : type
            object at the position of the given index
        """
        element = self.element_list[index]
        return element

    def get_latest_element(self):
        """
        Returns the latest added element. 

        Returns
        ----------
        element : type
            list element the tail pointed at one step ago
        """
        while len(self.element_list) == 0:
            pass
        index = self.tail_pos - 1
        if index < 0:
            index += self.max_len
        return self.get_element(index)

    def get_next_element(self):
        """
        Updates the index of head if needed and returns the element
        head is pointing at. 

        Returns
        ----------
        element : type
            list element the head is pointing at in the next step 
        """
        
        # update head index 
        head_next = self.head_pos + 1 
        if head_next >= self.max_len: # wird erst erreicht, wenn liste komplett gefüllt ist
            head_next = 0
            
        while head_next == self.tail_pos:
            pass                        # maybe add timeout 
        
        distance_h_t = self.get_distance_pointers(self.tail_pos, head_next) 
        if distance_h_t > self.max_len/2:
            self.head_pos = int(self.tail_pos + self.max_len/2) % self.max_len
            head_difference = self.get_distance_pointers(self.head_pos, head_next)
        else: 
            self.head_pos = head_next
            head_difference = 0 

        #print("Head val: ", self.get_element(self.head_pos)[0][0])
        #print("Jumps = ", head_difference)
        return head_difference, self.get_element(self.head_pos)

    def get_distance_pointers(self, first_pointer, second_pointer):
        """
        Calculates the distance between two pointers/indicies
        considering the movement direction of the pointers.
    
        Parameters
        ----------
        first_pointer : int
            index of the first element pointer 
        second_pointer : int
            index of the second element pointer 

        Returns
        ----------
        distance : int
            number of elements between the two pointers
        """
        
        if second_pointer > first_pointer: 
            return first_pointer - second_pointer + self.max_len
        return first_pointer - second_pointer

    def get_previous_elements(self, n):
        """
        Returns the last n frames before the head pointer. 

        Returns
        ----------
        element_list : list
            list of n elements before the head pointer 
        """
        
        distance_t_h = self.get_distance_pointers(self.head_pos, self.tail_pos) - 2
        if distance_t_h < n:
            n = distance_t_h
        element_list = []
        for i in range(self.head_pos - n, self.head_pos):
            element_list.append(self.get_element(i))
        return element_list

    def visualize_head_tail(self):
        """
        Prints a list representing the position of head and tail and
        a list showing the listed elements.
        """
        output_list = []
        for i in range(self.max_len):
            output_list.append("  #")
            if i == self.tail_pos:
                output_list[i] = "  T"
            if i == self.head_pos:
                output_list[i] = "  H"
        print(output_list)

        output_list = []
        for i in range(self.max_len):
            if i < len(self.element_list):
                output_list.append("{:3.0f}".format(self.element_list[i][0][0]))
        print(output_list)



def testLoop():
    for i in range(100):
        element = i* np.ones((5,1))
        buffer.append_element(element)
        buffer.visualize_head_tail()
        time.sleep(1)

if __name__ == '__main__':
    """ test script - add some example elements and get them from the buffer. """
    print("No test script available.")
    """
    buffer = RingBuffer()
    th = threading.Thread(target=testLoop)
    th.start()
    time.sleep(1)
    for i in range(100):
        jumps, element = buffer.get_next_element()
        # print(buffer.get_latest_element())
        time.sleep(2.1)
    """
