import numpy as np

def my_fit( words ):
    dt = Tree( min_leaf_size = 1, max_depth = 15 )
    dt.fit( words )
    return dt


class Tree:
    def __init__( self, min_leaf_size, max_depth ):
        self.root = None
        self.min_leaf_size = min_leaf_size
        self.max_depth = max_depth
    
    def fit( self, words ):
        self.root = Node( depth = 0, parent = None )
        mine_words = []
        for i in range(len(words)):
            mine_words.append([words[i], i])
            
        # The root is trained with all the words
        self.root.fit( my_words = mine_words, min_leaf_size = self.min_leaf_size, max_depth = self.max_depth, is_root = 1 )


class Node:
    # A node stores its own depth (root = depth 0), a link to its parent
    # A link to all the words as well as the words that reached that node
    # A dictionary is used to store the children of a non-leaf node.
    # Each child is paired with the response that selects that child.
    # A node also stores the query-response history that led to that node
    # Note: my_words_idx only stores indices and not the words themselves
    def __init__( self, depth, parent ):
        self.depth = depth
        self.parent = parent
        self.children = {}
        self.is_leaf = True
        self.query_idx = None
    
    # Each node must implement a get_query method that generates the
    # query that gets asked when we reach that node. Note that leaf nodes
    # also generate a query which is usually the final answer
    def get_query( self ):
        return self.query_idx
    
    # Each non-leaf node must implement a get_child method that takes a
    # response and selects one of the children based on that response
    def get_child( self, response ):
        # This case should not arise if things are working properly
        # Cannot return a child if I am a leaf so return myself as a default action
        if self.is_leaf:
            print( "Why is a leaf node being asked to produce a child? Melbot should look into this!!" )
            child = self
        else:
            # This should ideally not happen. The node should ensure that all possibilities
            # are covered, e.g. by having a catch-all response. Fix the model if this happens
            # For now, hack things by modifying the response to one that exists in the dictionary
            if response not in self.children:
                print( f"Unknown response {response} -- need to fix the model" )
                response = list(self.children.keys())[0]
            
            child = self.children[ response ]
            
        return child
    
    # Dummy leaf action -- just return the first word
    def process_leaf( self, my_words ):
        return my_words[0][1]
    
    def reveal( self, word, query ):
        # Find out the intersections between the query and the word
        mask = [ *( '_' * len( word ) ) ]
        
        for i in range( min( len( word ), len( query ) ) ):
            if word[i] == query[i]:
                mask[i] = word[i]
        
        return ' '.join( mask )
    
    # Dummy node splitting action -- use a random word as query
    # Note that any word in the dictionary can be the query
    def process_node( self, my_words, is_root ):
        # For the root we do not ask any query -- Melbot simply gives us the length of the secret word
        if is_root == 1:
            query_idx = -1
            query = ""
        else:
            max_g = 0
            
            k = min(25, len(my_words))
            # Adjust K 
            for word_pair in my_words[:k]:
                q = word_pair[0]
                m_num = {}
                
                for wrd in my_words:
                    mask = self.reveal( wrd[0], q )
                    if mask not in m_num:
                        m_num[ mask ] = 1
                    else :
                        m_num[ mask ] += 1

                val = np.array(list(m_num.values()))
                prob = val/np.sum(val)
                
                infogain_i = -np.sum(prob * np.log(prob)) / np.log(2)
                
                if(max_g <= infogain_i) :
                    max_g = infogain_i
                    query_idx = word_pair[1]
                    query = word_pair[0]

        
        split_dict = {}
        
        for wrd in my_words:
            mask = self.reveal( wrd[0], query )
            if mask not in split_dict:
                split_dict[ mask ] = []
            
            split_dict[ mask ].append( wrd )
        
        return ( query_idx, split_dict )
    
    def fit( self, my_words, min_leaf_size, max_depth, is_root, fmt_str = "    "):
        # self.my_words = my_words
        
        # If the node is too small or too deep, make it a leaf
        # In general, can also include purity considerations into account
        if len( my_words ) <= min_leaf_size or self.depth >= max_depth:
            self.is_leaf = True
            self.query_idx = self.process_leaf( my_words )

        else:
            self.is_leaf = False
            ( self.query_idx, split_dict ) = self.process_node( my_words, is_root)
            
            for ( i, ( response, split ) ) in enumerate( split_dict.items() ):
                # Create a new child for every split
                self.children[ response ] = Node( depth = self.depth + 1, parent = self )
                
                # Recursively train this child node
                self.children[ response ].fit( split, min_leaf_size, max_depth, 0, fmt_str)