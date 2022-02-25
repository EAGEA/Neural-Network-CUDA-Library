//
// Created by Emilien Aufauvre on 12/01/2022.
//

#ifndef CUDANN_GLOBAL_H
#define CUDANN_GLOBAL_H


/**
 * Execute on device or host.
 */
#define _USE_GPU true


/**
 * Show logs.
 */
#define _DEBUG false
#define _ERROR true

/**
 * Max number of thread in a block (to be set depending on GPU).
 */
#define MAX_NB_THREADS_BLOCK 1024


#endif //CUDANN_GLOBAL_H