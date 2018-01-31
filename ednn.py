import tensorflow as tf


class EDNN_helper(object):
    
    def __init__(self, L, f, c):
        assert f <= L/2, "Focus must be less than half the image size to use this implementation."
        assert (f + 2*c) <= L, "Total tile size (f+2c) is larger than input image."
        self.l = L
        self.f = f
        self.c = c
    
    def __roll(self, in_, num, axis):
        """author: Kyle Mills"""
        D = tf.transpose(in_, perm=[axis, 1-axis])  #if axis=1, transpose first
        D = tf.concat([D[num:, :], D[0:num, :]], axis=0)
        return tf.transpose(D, perm=[axis, 1-axis]) #if axis=1, transpose back

    def __slice(self, in_, x1, y1, w, h): 
        """author: Kyle Mills"""
        return in_[x1:x1+w, y1:y1+h]

    def ednn_split(self, in_): 
        """author: Kyle Mills"""
        l = self.l
        f = self.f
        c = self.c

        tiles = []
        for iTile in range(l/f):
            for jTile in range(l/f):
                #calculate the indices of the centre of this tile (i.e. the centre of the focus region)
                cot = (iTile*f + f/2, jTile*f + f/2) #centre of tile
                foc_centered = in_ 
                #shift the picture, wrapping the image around,
                #so that the focus is centered in the middle of the image
                foc_centered = self.__roll(foc_centered, l/2-cot[0], 0)
                foc_centered = self.__roll(foc_centered, l/2-cot[1], 1)
                #Finally slice away the excess image that we don't want to appear in this tile
                final = self.__slice(foc_centered, l/2-f/2-c, l/2-f/2-c, 2*c+f, 2*c+f)
                tiles.append(final)

        return tf.expand_dims(tiles, axis=3)
