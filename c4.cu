#include <stdio.h>
#include <sys/time.h>
#include <cuda.h>
#include <fstream>
#include <iostream>


#define N_ROWS 5
#define N_COLUMNS 6
#define INF 99999
#define K 60000000
#define SERIAL_DEPTH 10
#define GPU_DEPTH 2

#define at(table, i, j) ((table[1] & (1LL << ((i) * N_COLUMNS + j))) ? ( ((table[0] & (1LL << ((i) * N_COLUMNS + j)))!=0LL) + 1 ) : 0)

// error checking for CUDA calls: use this around ALL your calls!
#define GPU_CHECKERROR( err ) (gpuCheckError( err, __FILE__, __LINE__ ))
static void gpuCheckError( cudaError_t err,
                         const char *file,
                         int line ) {
   if (err != cudaSuccess) {
       printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
              file, line );
       exit( EXIT_FAILURE );
   }
}

__host__ __device__ void set_at(long long int table[2], int i, int j, int x) {
    if(x==2)
        table[0] |= (1LL << (i * N_COLUMNS + j));
    table[1] |= (1LL << (i * N_COLUMNS + j));
}

int table_size = N_ROWS * N_COLUMNS;
long long int table[2] = {0LL, 0LL};

// Either 1 or 2
int current_player = 1;
int max_player = 1;
// Each player's score
int score1 = 0;
int score2 = 0;
// current column picked by current player
int current_move;
// current row picked by current player
int current_row;

/*
 Prints table and score
 */
void print_table(long long int t[2]) {
    
    printf("~* CONNECT 4 *~ \n \n");
    
    // Print table
    for (int i = 0; i < N_ROWS; i++) {
        printf("|");
        for (int j = 0; j < N_COLUMNS; j++) {
            if (at(t, i, j) == 0)
                printf(" . ");
            if (at(t, i, j) == 1)
                printf(" X ");
            if (at(t, i, j) == 2)
                printf(" 0 ");
            printf("|");
        }
        printf("\n");
    }
    
    // Print numbers
    printf("\n+ ");
    for (int j=0; j < N_COLUMNS; j++)
        printf("%d   ",j);
    printf("+ \n \n");
    
    //     Score
    printf("SCORE: \n Player 1 (X, Human) = %d \n Player 2 (0, Computer) = %d \n \n", score1, score2);
}

/*
 Checks if player won by making a sequence of 4 markers either
 horizontally, vertically or diagonally.
 */
__device__ __host__ int current_player_won(long long int table[2], int current_row, int current_move, int current_player){
    // Check for vertical sequence
    // Look at last marker placed and compare with the 3 markers below it
    if ((current_row < N_ROWS-3)
        && (at(table,current_row,current_move) == at(table,current_row + 1,current_move))
        && (at(table,current_row + 1,current_move) == at(table,current_row + 2,current_move))
        && (at(table,current_row+ 2,current_move) == at(table,current_row + 3,current_move)))
        return true;
    
    // Check for horizontal sequence
    int sequence_length = 1;
    int j = 1;
    while ((current_move - j >= 0) && (at(table,current_row,current_move - j) == current_player)){
        j++; sequence_length++;
    }
    j = 1;
    while ((current_move + j < N_COLUMNS) && (at(table,current_row,current_move + j) == current_player)){
        j++; sequence_length++;
    }
    if (sequence_length >= 4)
        return true;
    
    //Check for diagonal sequence
    sequence_length = 1;
    j = 1;
    while((current_move - j >= 0) && (current_row - j >= 0) && (at(table,current_row - j,current_move - j) == current_player)){
        j++; sequence_length++;
    }
    j = 1;
    while ((current_move + j < N_COLUMNS) && (current_row + j <= 5) && (at(table,current_row + j,current_move + j) == current_player)){
        j++; sequence_length++;
    }
    if (sequence_length >= 4)
        return true;
    
    //Check for inverted diagonal sequence
    sequence_length = 1;
    j = 1;
    while((current_move - j >= 0) && (current_row + j < N_ROWS) && (at(table,current_row + j,current_move - j) == current_player)){
        j++; sequence_length++;
    }
    j = 1;
    while ((current_move + j < N_COLUMNS) && (current_row - j >= 0) && (at(table,current_row - j,current_move + j) == current_player)){
        j++; sequence_length++;
    }
    if (sequence_length >= 4)
        return true;
    
    return false;
}


__device__ __host__ int column_is_full (long long int table[2], int column_j) {
    return (at(table, 0, column_j) != 0);
}

__device__ __host__ int table_is_full(long long int table[2]) {
    for (int j = 0; j < N_COLUMNS; j++){
        //If some column is not full, then table is not full
        if (at(table,0,j) == 0)
            return false;
    }
    return true;
}

/*
 Structures for maintaining the state of the recursion.
 */
typedef struct state
{
    long long int table[2]; //board state
    int current_move;
    int parent_index;
    int node_value;
    int child_count; // -1 means children havent been generated yet, non negative numbers mean how many children are left to be checked for minmax values
    int depth;
    
} state;


__device__ __host__ state new_state(long long int t[2], int current_move, int parent_index, int node_value, int child_count, int depth){
    state s;
    s.table[0] = t[0];
    s.table[1] = t[1];
    s.current_move = current_move;
    s.parent_index = parent_index;
    s.node_value = node_value;
    s.child_count = child_count;
    s.depth = depth;
    
    return s;
}

__device__ __host__ void print_state(state s){
    // print_table(s.table);
    printf("current move %d\n", s.current_move);
    printf("parent index %d\n", s.parent_index);
    printf("node value %d\n", s.node_value);
    printf("child count %d\n", s.child_count);
    printf("depth %d\n", s.depth);
}

typedef struct stack
{
    int last_i;
    state data[600];
} stack;

__device__ __host__ void stack_push(stack &s, state some_state){
    s.last_i = s.last_i + 1;
    s.data[s.last_i] = some_state;
}

__device__ __host__ int stack_is_empty(stack &s){
    return (s.last_i == -1);
}

__device__ __host__ state stack_pop(stack &s){
    state st = s.data[s.last_i];
    s.last_i--;
    return st;
}

__device__ __host__ state stack_peek(stack &s){
    // printf("last i: %d\n",s.last_i);
    return s.data[s.last_i];
}

__device__ __host__ stack new_stack(){
    stack s;
    s.last_i = -1;
    return s;
}

int k = -1;
// For storing the k nodes in SERIAL DEPTH 
long long int * kTables = new long long int [K * 2];
// For getting the k values back from the GPU
int * kValues = new int [K];

void add_table(long long int t[2]){
    k++;
    kTables[k*2] = t[0];
    kTables[k*2+1] = t[1];
}

int retrieve_value(){
    int value = kValues[k];
    k--;
    return value;
}

// Minmax that runs on the GPU
// Return value of the origin node
__device__ __host__ int d_minmax(long long int current_table[2], int origin_is_max){
    
    int value = INF;
    if (origin_is_max) value = -INF;
    
    state origin;
    origin = new_state(current_table, -1, -1, value, -1, 0);
    
    stack s = new_stack();
    
    stack_push(s, origin);
    
    // int best_move = -1;
    
    int final_value = -1;
    
    while(!stack_is_empty(s)){
        state current_state = stack_peek(s);
        int current_index = s.last_i;
        int current_depth = current_state.depth;
        int is_max = (current_depth % 2 == 0);
        if (!origin_is_max) is_max = !(is_max);
        int last_player = 1;
        if (is_max) last_player = 2;
        int current_move = current_state.current_move;
        int parent_index = current_state.parent_index;
        
        //  printf("current state:\n");
        //  print_state(current_state);
        
        int lose = 0, full_table = 0, all_children_accounted_for = 0;
        // If not the origin node, then test for win and full table situations
        if (current_move != -1){
            // retrieve row index of current state's move
            int row = 0;
            while (at(current_state.table,row, current_move) == 0)
                row++;
            
            //check if player wins in this move
            lose = current_player_won(current_state.table, row, current_move, last_player);
            
            //check if table gets full
            full_table = table_is_full(current_state.table);
        }
        
        all_children_accounted_for = current_state.child_count == 0;
        
        int at_max_depth = (current_depth == GPU_DEPTH);

        //if node is terminal (leaf) or all children have been computed
        if (lose || full_table || all_children_accounted_for || at_max_depth){//
            
            
            int value = 0;
            if (lose)
                value = -1;
            if (!is_max)
                value = -value;
            if (all_children_accounted_for)
                value = current_state.node_value;
            
            // If origin node, then end search
            if (current_move == -1){
                final_value = value;
                stack_pop(s);
                continue;
            }
            
            state parent_state = s.data[parent_index];
            int parent_value = parent_state.node_value;
            
            int parents_parent_index = s.data[parent_index].parent_index;
            
            //If current state is max, the parent state is min
            if(is_max){
                // if parent state has bigger value, give it the smaller value
                if (parent_value > value){
                    
                    s.data[parent_index].node_value = value;
                    // if(current_depth == 1)
                    //     best_move = current_move;
                    
                    if (parents_parent_index != -1){
                        int alpha = s.data[parents_parent_index].node_value;
                        int beta = value;
                        if (alpha >= beta){
                            s.last_i = parent_index;
                            s.data[parent_index].child_count = 0;
                            continue;
                        }
                    }
                    
                }
            }//otherwise, parent state is max
            else{
                if (parent_value < value){
                    
                    s.data[parent_index].node_value = value;
                    // if(current_depth == 1)
                    //     best_move = current_move;
                    
                    if (parents_parent_index != -1){
                        int beta = s.data[parents_parent_index].node_value;
                        int alpha = value;
                        if (alpha >= beta){
                            s.last_i = parent_index;
                            s.data[parent_index].child_count = 0;
                            continue;
                        }
                    }
                    
                }
            }
            
            s.data[parent_index].child_count--;
            stack_pop(s);
            continue;
        }
        
        // Generate children
        int child_count = 0;
        int child_value = -INF;
        if (is_max) child_value = INF;
        
        for(int j = 0; j < N_COLUMNS; j++ ){
            if (column_is_full(current_state.table, j)) continue;
            child_count++;
            
            long long int child_table[2];
            child_table[0] = current_state.table[0];
            child_table[1] = current_state.table[1];
            
            // Find row where we can play next
            int row = N_ROWS-1;
            while (at(child_table,row, j) != 0)
                row--;
            
            if (is_max)
                set_at(child_table, row, j, 1);
            else
                set_at(child_table, row, j, 2);
            
            state child;
            child = new_state(child_table, j, current_index, child_value, -1, current_depth+1);
            
            stack_push(s, child);
        }
        
        s.data[current_index].child_count = child_count;
    }
    
    // printf("best move %d\n", best_move);
    return final_value;
}

// Kernel function
// Receives all tables on the SERIAL DEPTH level of the tree and returns all the computed values
__global__ void minmax_gpu (long long int * tables, int * values, int origin_is_max, int k)
{
       int x = blockDim.x * blockIdx.x + threadIdx.x;
       if (x > k) return;
    
       long long int table[2];
       table[0] = tables[x*2];
       table[1] = tables[x*2+1];

       int value = d_minmax(table, origin_is_max);
       values[x] = value;
}

// CPU minmax. Called twice per turn, once to store all the nodes in the SERIAL DEPTH level of the tree
// and another one after the GPU part has returned with the computer values for those nodes
__host__ int h_minmax(long long int current_table[2], int origin_is_max, int first_pass){
    int breadth=0;
    
    int value = INF;
    if (origin_is_max) value = -INF;
    
    state origin;
    origin = new_state(current_table, -1, -1, value, -1, 0);
    
    stack s = new_stack();
    
    stack_push(s, origin);
    
    int best_move = -1;
    
    if (first_pass){
        k = -1;
    }
    
    while(!stack_is_empty(s)){
        state current_state = stack_peek(s);
        int current_index = s.last_i;
        int current_depth = current_state.depth;
        int is_max = (current_depth % 2 == 0);
        if (!origin_is_max) is_max = !(is_max);
        int last_player = 1;
        if (is_max) last_player = 2;
        int current_move = current_state.current_move;
        int parent_index = current_state.parent_index;
        
        //  printf("current state:\n");
        //  print_state(current_state);
        
        int lose = 0, full_table = 0, all_children_accounted_for = 0;
        // If not the origin node, then test for win and full table situations
        if (current_move != -1){
            // retrieve row index of current state's move
            int row = 0;
            while (at(current_state.table,row, current_move) == 0)
                row++;
            
            //check if player wins in this move
            lose = current_player_won(current_state.table, row, current_move, last_player);
            
            //check if table gets full
            full_table = table_is_full(current_state.table);
        }
        
        all_children_accounted_for = current_state.child_count == 0;
        
        int second_pass_leaf = (current_depth == SERIAL_DEPTH && !first_pass);
        
        //if node is terminal (leaf) or all children have been computed
        if (lose || full_table || all_children_accounted_for || second_pass_leaf){//
            
            // If origin node, then end search
            if (current_move == -1){
                stack_pop(s);
                continue;
            }
            
            int value = 0;
            if (lose){
                value = -1;
                if (!is_max)
                    value = -value;
            }
            else if (all_children_accounted_for)
                value = current_state.node_value;
            else if (second_pass_leaf){
                value = retrieve_value();
                               // printf("value %d\n", value);
            }
            state parent_state = s.data[parent_index];
            int parent_value = parent_state.node_value;
            
            //If current state is max, the parent state is min
            if(is_max){
                // if parent state has bigger value, give it the smaller value
                if (parent_value > value){
                    
                    s.data[parent_index].node_value = value;
                    if(current_depth == 1)
                        best_move = current_move;
                    
                }
            }//otherwise, parent state is max
            else{
                if (parent_value < value){
                    
                    s.data[parent_index].node_value = value;
                    if(current_depth == 1)
                        best_move = current_move;
                    
                }
            }
            
            s.data[parent_index].child_count--;
            stack_pop(s);
            continue;
        }
        
        
        if (current_depth == SERIAL_DEPTH && first_pass){
            add_table(current_state.table);
            breadth++;
            //            printf("breadth %d\n", breadth);
            stack_pop(s);
            s.data[parent_index].child_count--;
            continue;
        }
        
        // Generate children
        int child_count = 0;
        int child_value = -INF;
        if (is_max) child_value = INF;
        
        for(int j = 0; j < N_COLUMNS; j++ ){
            if (column_is_full(current_state.table, j)) continue;
            child_count++;
            
            long long int child_table[2];
            child_table[0] = current_state.table[0];
            child_table[1] = current_state.table[1];
            
            // Find row where we can play next
            int row = N_ROWS-1;
            while (at(child_table,row, j) != 0)
                row--;
            
            if (is_max)
                set_at(child_table, row, j, 1);
            else
                set_at(child_table, row, j, 2);
            
            state child;
            child = new_state(child_table, j, current_index, child_value, -1, current_depth+1);
            
            stack_push(s, child);
        }
        
        s.data[current_index].child_count = child_count;
    }
    
    // printf("best move %d\n", best_move);
    // printf("breadth0 %d\n", k);
    return best_move;
}

long long int *d_tables;
int *d_values;
// Main min max, encapsulates all the CPU and GPU dymanics
int minmax(long long int current_table[2], int origin_is_max){
    
    // Computer tables at SERIAL DEPTH level
    printf("First CPU pass \n");
    h_minmax(current_table, origin_is_max, 1);
    printf("Number of nodes found at SERIAL DEPTH level %d\n", k+1);
    
    int is_max = (SERIAL_DEPTH % 2 == 0);
    if (!origin_is_max) is_max = !(is_max);
    
    if (k > -1){
        // printf("Bytes sent to GPUK*2 * sizeof (long long int) %lu\n", (K*2 * sizeof (long long int)));
        printf("K*2 * sizeof (long long int) %lu\n", (K*2 * sizeof (long long int)));
        printf("k*2 * sizeof (long long int) %lu\n", (k*2 * sizeof (long long int)));

        GPU_CHECKERROR(
        cudaMemcpy ((void *) d_tables,
                    (void *) kTables,
                    (k+1)*2 * (sizeof(long long int)),
                    cudaMemcpyHostToDevice)
        );
    
        unsigned int threads_per_block = 512;
        unsigned int num_blocks = ceil ((k+1) / (1.0*threads_per_block) );
    
        // launch the kernel:
        printf("Launching Kernel \n");
        minmax_gpu<<<num_blocks, threads_per_block>>>
                                            (d_tables,
                                            d_values,is_max,k);
    
        // get back the move:
        GPU_CHECKERROR(
        cudaMemcpy ((void *) kValues,
                    (void *) d_values,
                    K*sizeof(int),
                    cudaMemcpyDeviceToHost)
        );
    
        // make sure the GPU is finished doing everything!
        GPU_CHECKERROR(
            cudaDeviceSynchronize()
        );

        printf( "Errors?: %s \n", cudaGetErrorString(cudaPeekAtLastError()));
   
    }
    
    printf("Second CPU pass \n");
    int move = h_minmax(current_table, origin_is_max, 0);
    printf("Best move computed: %d\n",move);
    // printf("k %d\n", k);
    
    return move;
}

void clear_table(){
    table[0]= 0LL;
    table[1]= 0LL;
}

/*
 Ask player which column to pick and change table accordingly
 */
void pick_column() {
    
    if (current_player == 1){
        current_move = -1;
        printf("Pick a column, then press enter, player %d:\n", current_player);
        scanf ("%d", &current_move);
        
        // while move is invalid, keep asking
        while(current_move < 0 || current_move > N_COLUMNS-1 || column_is_full(table, current_move)){
            printf("invalid move, pick another column:\n");
            scanf ("%d",&current_move);
        }
    }else{
        current_move = minmax(table,max_player==2);
    }
    
    // Find row where his move will be performed
    // It will be the first row with value equals 0
    int row = N_ROWS-1;
    while (at(table,row,current_move) != 0)
        row--;
    
    // Change table accordingly
    set_at(table, row, current_move, current_player);
    
    // Store row where player just placed his marker
    // Used to check if current player won the game
    current_row = row;
}

/*
 Switch current player
 */
void switch_player(){
    if (current_player == 1)
        current_player = 2;
    else
        current_player = 1;
}

/*
 Increase current player's score
 */
void update_score(){
    if (current_player == 1)
        score1++;
    else
        score2++;
}


int main (int argc, char *argv[])
{
    GPU_CHECKERROR(
        cudaMalloc ((void **) &d_tables, K * 2 * sizeof (long long int))
    );
    GPU_CHECKERROR(
        cudaMalloc ((void **) &d_values, K * sizeof (int))
    );
    
    // game loop
    while (true) {
        clear_table();
        print_table(table);
        // The player who starts is the max player
        max_player = current_player;
        
        // match loop
        while (true) {
            pick_column();
            print_table(table);
            
            if (current_player_won(table, current_row, current_move, current_player)){
                update_score();
                print_table(table);
                printf("Congratulations, Player %d! \n", current_player);
                // leave match
                break;
            }
            // If nobody wins and table is full, we come to a draw
            else if (table_is_full(table)){
                printf("Draw! \n");
                // leave match
                break;
            }
            
            switch_player();
        }
        
        printf("Do you wish to play again?(y/n) \n");
        char play_again;
        scanf (" %c",&play_again);
        if (!(play_again == 'y' || play_again == 'Y'))
            break;
        
    }
    
    cudaFree (d_tables);
    cudaFree (d_values);
    delete[] kValues;
    delete[] kTables;
    return 0;
    
}