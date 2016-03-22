#ifndef EVENT_POOL_HEADER
#define EVENT_POOL_HEADER

#include "util.hpp"

/******************************************************************************************

    This class handles the dependence relationship between events.
    Consider the following CNN with 6 layers and 4 inputs.

    Lengend:
        +           one event specified as (layerId, inId)
        + ---> +    the right event is dependent on the left one

                    LayerId 0 - 5
        + ---> + ---> + ---> + ---> + ---> +
             /      /      /      /      /
            /      /      /      /      /
           /      /      /      /      /
          /      /      /      /      /
        + ---> + ---> + ---> + ---> + ---> +
             /      /      /      /      /
 inId       /      /      /      /      /
 0 - 3     /      /      /      /      /
          /      /      /      /      /
        + ---> + ---> + ---> + ---> + ---> +
             /      /      /      /      /
            /      /      /      /      /
           /      /      /      /      /
          /      /      /      /      /
        + ---> + ---> + ---> + ---> + ---> +

    All the events are stored as vector<vector<cl_event>>, mapped as:
        
        + ---> + ---> + ---> + ---> + ---> +
                    /      /      /      /  
                   /      /      /      /   
                  /      /      /      /    
                 /      /      /      /     
               + ---> + ---> + ---> + ---> + ---> +
                           /      /      /      /               
                          /      /      /      /                
                         /      /      /      /                 
                        /      /      /      /                  
                      + ---> + ---> + ---> + ---> + ---> +
                                  /      /      /      /  
                                 /      /      /      /   
                                /      /      /      /    
                               /      /      /      /     
                             + ---> + ---> + ---> + ---> + ---> +
                                    

    cluster 0   : (0, 0)
    cluster 1   : (1, 0)
    cluster 2   : (2, 0), (0, 1)
    cluster 3   : (3, 0), (1, 1)
    cluster 4   : (4, 0), (2, 1), (0, 2)
    cluster 5   : (5, 0), (3, 1), (1, 2)
    cluster 6   : (4, 1), (2, 2), (0, 3)
    cluster 7   : (5, 1), (3, 1), (1, 3)
    cluster 8   : (4, 2), (2, 3)
    cluster 9   : (5, 2), (3, 3)
    cluster 10  : (4, 3)
    cluster 11  : (5, 3)


*******************************************************************************************/
class EventPool {
public:

    // Constructor.
    EventPool(size_t layerNum, size_t inNum) : layerNum(layerNum), inNum(inNum) {
        itemNum = (layerNum + 1) / 2;
        clusterNum = 2 * (inNum - 1) + layerNum;
        pool.resize(clusterNum, std::vector<cl_event>(itemNum));
    }
    ~EventPool() {}

    /// Given a specific layer in one input, get the cl_event list it is dependent on.
    /// len is used to return the length of the event list (usually 1 or 2).
    /// Notice for an event with (layerId, inId), we have the formular:
    ///
    /// (0, 0)                  -> NONE;                                            0
    /// (layerId, 0)            -> (layerId - 1, 0);                                1
    /// (0, inId)               -> (1, inId - 1);                                   1
    /// (layerNum - 1, inId)    -> (layerNum - 2, inId);                            1
    /// (layerId, inId)         -> (layerId + 1, inId - 1), (layerId - 1, inId);    2
    cl_event *getDependentEventList(size_t layerId, size_t inId, uint32_t *len) {
        if (layerId == 0 && inId == 0) {
            *len = 0;
            return nullptr;
        }
        else if (inId == 0) {
            *len = 1;
            size_t cluster = getClusterId(layerId - 1, 0);
            size_t item = getItemId(layerId - 1, 0);
            return &(pool[cluster][item]);
        }
        else if (layerId == 0) {
            *len = 1;
            size_t cluster = getClusterId(1, inId - 1);
            size_t item = getItemId(1, inId - 1);
            return &(pool[cluster][item]);
        }
        else if (layerId == layerNum - 1) {
            *len = 1;
            size_t cluster = getClusterId(layerNum - 2, inId);
            size_t item = getItemId(layerNum - 2, inId);
            return &(pool[cluster][item]);
        }
        else {
            *len = 2;
            size_t cluster = getClusterId(layerId + 1, inId - 1);
            size_t item = getItemId(layerId + 1, inId - 1);
            return &(pool[cluster][item]);
        }
    }

    // Push one cl_event into the pool.
    // Notice that cl_event is a pointer, so we can pass by value.
    void pushEvent(size_t layerId, size_t inId, cl_event event) {
        size_t cluster = getClusterId(layerId, inId);
        size_t item = getItemId(layerId, inId);
        pool[cluster][item] = event;
    }

    // Return a new event list, but sorted by (inId * layerNum + layerId).
    std::vector<cl_event> sort() const {
        std::vector<cl_event> sorted;
        sorted.reserve(layerNum * inNum);
        for (int inId = 0; inId < inNum; ++inId) {
            for (int layerId = 0; layerId < layerNum; ++layerId) {
                size_t cluster = getClusterId(layerId, inId);
                size_t item = getItemId(layerId, inId);
                sorted.push_back(pool[cluster][item]);
            }
        }
        return sorted;
    }
    
    size_t layerNum, inNum;
    size_t clusterNum, itemNum;
    std::vector<std::vector<cl_event>> pool;

    // Get the cluster id.
    inline size_t getClusterId(size_t layerId, size_t inId) const {
        return layerId + inId * 2;
    }

    // Get the item id.
    inline size_t getItemId(size_t layerId, size_t inId) const {
        size_t a = (layerNum - 2 - layerId + 1) / 2;
        return a > inId ? inId : a;
    }

};

#endif