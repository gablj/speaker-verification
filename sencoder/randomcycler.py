import random 
from typing import List, Any 

class RandomCycler: 
    """
    The purpose of this class is to allow access to the items of a sequence in
    a constrained random order. This class provides a way to shuffle a sequence
    and repeatedly sample items from it in a way that guarantees that each item
    is sampled a certain number of times, while still preserving some degree of randomness.
    This class creates an internal copy of a sequence and allows to access to its items
    in a random order that is constrained in the following way: 
    -for a source sequence of 'n' items and one or several consecutive queries of a total of 'm' items
    (total number of items sampled), each item in the sequence is guaranteed to be returned between 
    'm // n' and '((m - 1) // n) + 1' times, the exact number of times each item is returned will
    be determined randomly, but it will fall within this range.
    -Also, between two apperances of the same item, there may be at most '2 * (n - 1)' other items.

    Attributes
    ----------
    all_items : List 
        A list containing all the items that will be sampled.
    next_items : List 
        A list that serves as a buffer to hold the next set of items to be sampled.
    
    Methods
    -------
    sample(count: int) -> List
        Samples 'count' number of items from the sequence in a constrained random order.

    next() -> Any
        To implement python's built in 'next()' function bahaviour. This method allows 
        the class to be used as an iterator. It samples the next item from the sequence
        using the 'sample()' method. The method returns the first item from the sampled list.
    """
    def __init__(self, source: List):
        """
        Initializes the class by creating a copy of the input sequence and
        an empty list to hold the next items to be sampled. 

        Parameters
        ----------
        source : List 
            A list containing the elements to be sampled. 
        """
        if len(source) == 0:
            raise Exception("Empty collection")
        self.all_items = source
        self.next_items = []    #To keep track of the next set of items to be returned by the sample() method

    def sample(self, count: int) -> List:
        """
        Samples 'count' number of items in a random constrained order:
        -for a source sequence of 'n' items and one or several consecutive queries of a total of 'm' items
        (total number of items sampled), each item in the sequence is guaranteed to be returned between 
        'm // n' and '((m - 1) // n) + 1' times, the exact number of times each item is returned will
        be determined randomly, but it will fall within this range.
        -between two apperances of the same item, there may be at most '2 * (n - 1)' other items.
        
        The class instance maintains an internal buffer called 'next_items', which holds the next set of items to be sampled.
        If there are enough items in the buffer to fulfill the 'count' requirement, they are taken from the buffer.
        If not, the buffer is refilled with randomly shuffled items from the sequence.
        
        Parameters
        ----------
        count : int 
            The total number of items to be sampled.

        Returns
        -------
        List
            A list of length 'count' that contains the sampled items from the sequence.
        """
        shuffle = lambda l: random.sample(l, len(l))

        out = []
        #i = 0; j =0                      #Remove in final version,counter added for the testing script
        while count > 0: 
            if count >= len(self.all_items):
                #i+= 1                    #Remove in final version, added for the testing script         
                #print("**** i = %d" % i) #Remove in final version, added for the testing script
                out.extend(shuffle(list(self.all_items)))
                count -= len(self.all_items)
                continue
            n = min(count, len(self.next_items))
            out.extend(self.next_items[:n])
            #j += 1                        #Remove in final version, added for the testing script
            #print("j =", j,"out = ", out, "next_items = ", self.next_items) #Remove in final version, added for the testing script
            count -= n
            self.next_items = self.next_items[n:]
            if len(self.next_items) == 0: #At the first iteration "next_items" is empty
                #print("***** len(self.next_items) = %d" % len(self.next_items)) #Remove in final version, added for the testing script
                self.next_items = shuffle(list(self.all_items))
        return out

    def __next__(self) -> Any:
        '''
        To implement python's built in 'next()' function bahaviour. This method allows 
        the class to be used as an iterator. It samples the next item from the sequence
        using the 'sample()' method. The method returns the first item from the sampled list.

        Returns
        -------
        Any
            The next sampled item from the sequence. In the context of this project
            is used to return items of class 'Utterance' type or 'Speaker' type. 
        '''
        return self.sample(count=1)[0] 

"""
#A testing script, remove in final version

a = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
rc = RandomCycler(a)
n = len(a)
m = 3 * n + 5
print("Before sampling: rc.all_items = ", rc.all_items, "rc.next_items = ", rc.next_items)
print("n = %d, m = %d, sample limits min:m // n = %d and max:((m - 1) // n) + 1 = %d \n" % (n, m, m // n, ((m - 1) // n) + 1))
sample = rc.sample(m)
print("Samplig m = %d items: sample(m) = %r, len(sample) = %d \n" % (m, sample, len(sample)) )
print("Sampling a single item: next(rc) = ", next(rc))
"""
