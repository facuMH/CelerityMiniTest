#include <cstdlib>
#include <cstdio>
#include <random>
#include <vector>
#include <chrono>
#include <iostream>
#include <fstream>

#include <celerity/celerity.h>

const float twoPi = 3.14159265358979323846264338328f*2.0f;

auto rands = std::uniform_real_distribution<float>(0.0f, 1.0f);

static float myRand(size_t index) {
    return rands(index);
}

static void corners(celerity::distr_queue& queue,
                    const celerity::buffer<float, 2> mx_buf,
                    celerity::buffer<float, 2> new_buff,
                    const int N){

    queue.submit(
        [=](celerity::handler cgh) {
            auto mx_dev = mx_buf.get_access<cl::sycl::access::mode::read>(cgh,
                                    celerity::accessor::neighborhood<2>(1,1));
            auto new_dev = new_buf.get_access<cl::sycl::access::mode::
                                    discard_write>(cgh, celerity::accessor::one_to_one<2>());

            cgh.host_task(
                celerity::on_master_node,
                [=]() {
                    
                    float theta_n[4], phi_n[4], sen_theta[4];
                    float new_x[4], delta[4], E[4];

                    cl::sycl::item<2> items[4];
                    size_t idxs[4];

                    items[0] = {0, 0};
                    idxs[0] = 0;

                    items[1] = {N-1, 0};
                    idxs[1] = N-1;

                    items[2] = {0, N-1};
                    idxs[2] = N*(N-1);

                    items[3] = {N-1, N-1};
                    idxs[4] = N*N-1;

                    for (int i=0; i<4; i++){

                        theta_n[i] = acosf(2.0f*(0.5f - myRand(idxs[i])));
                        phi_n[i] = twoPi*myRand(idxs[i]);
                        sen_theta[i] = sinf(theta_n[i]);

                        new_x[i] = sen_theta[i]*cosf(phi_n[i]);
                        delta[i] = new_x[i] - mx_dev[items[i]];
                    }

                    E[0] += mx_dev[{1, 0}]*delta[0];
                    E[0] += mx_dev[{0, 1}]*delta[0];

                    E[1] = mx_dev[{N-2, 0}]*delta[1];
                    E[1] += mx_dev[{N-1, 1}]*delta[1];

                    E[2] = mx_dev[{0, N-2}]*delta[2];
                    E[2] += mx_dev[{1, N-1}]*delta[2];

                    E[3] = mx_dev[{N-1, N-2}]*delta[3];
                    E[3] += mx_dev[{N-2, N-1}]*delta[3];

                    for (int i=0; i<4; i++) {
                        if (E[i] < 0.0f) {
                            new_dev[items[i]] = new_x[i];
                        } else {
                            new_dev[items[i]] = mx_dev[items[i]];
                        }
                    }

                }
            );
        }
    );

}


static void sides(celerity::distr_queue& queue,
                 const celerity::buffer<float, 2>& mx_buf,
                 celerity::buffer<float, 2>& new_buff,
                 const int N) { 

    queue.submit(
        [=](celerity::handler& cgh) {
        
            auto mx_dev = mx_buf.get_access<cl::sycl::access::mode::read>(cgh,
                                    celerity::accessor::neighborhood<2>(1,1));
            auto new_dev = new_buf.get_access<cl::sycl::access::mode::
                                    discard_write>(cgh, celerity::accessor::one_to_one<2>());

            cgh.parallel_for<class Side>(
                cl::sycl::range<2>(N-2, 1),
                cl::sycl::id<2>(1,0),
                [=](cl::sycl::item<2> item){
  
                    float theta_n[4], phi_n[4], sen_theta[4];
                    float new_x[4], delta[4], E[4];

                    cl::sycl::item<2> items[4];
                    size_t idxs[4];

                    items[0] = {item[0], 0};
                    idxs[0] = item[0];

                    items[1] = {item[0], N-1};
                    idxs[1] = N*(N-1)+ item[0];

                    items[2] = {0, item[0]};
                    idxs[2] = N*item[0];

                    items[3] = {N-1, item[0]};
                    idxs[4] = N*item[0]+N-1;
               
                    for (int i=0; i<4; i++){

                        theta_n[i] = acosf(2.0f*(0.5f - myRand(idxs[i])));
                        phi_n[i] = twoPi*myRand(idxs[i]);
                        sen_theta[i] = sinf(theta_n[i]);

                        new_x[i] = sen_theta[i]*cosf(phi_n[i]);
                        delta[i] = new_x[i] - mx_dev[items[i]];

                    }
                
                    E[0] = mx_dev[{item[0]-1, 0}]*delta[0];
                    E[0] += mx_dev[{item[0]+1, 0}]*delta[0];
                    E[0] += mx_dev[{item[0], 1}]*delta[0];

                    E[1] = mx_dev[{item[1]-1, N-1}]*delta[1];
                    E[1] += mx_dev[{item[1]+1, N-1}]*delta[1];
                    E[1] += mx_dev[{item[1], N-2}]*delta[1];

                    E[2] = mx_dev[{0, item[0]-1}]*delta[2];
                    E[2] += mx_dev[{0, item[0]+1}]*delta[2];
                    E[2] += mx_dev[{1, item[0]}]*delta[2];

                    E[3] = mx_dev[{N-1, item[0]-1}]*delta[3];
                    E[3] += mx_dev[{N-1, item[0]+1}]*delta[3];
                    E[3] += mx_dev[{N-2, item[0]}]*delta[3];

                    for (int i=0; i<4; i++) {
                        if (E[i] < 0.0f) {
                            new_dev[items[i]] = new_x[i];
                        } else {
                            new_dev[items[i]] = mx_dev[items[i]];
                        }
                    }
                }
            );
        }
    );
}

static void update(celerity::distr_queue& queue,
                 const celerity::buffer<float, 2> mx_buf,
                 celerity::buffer<float, 2> new_buff,
                 const int N){
    queue.submit(

        [=](celerity::handler& cgh) {

            auto mx_dev = mx_buf.get_access<cl::sycl::access::mode::read>(cgh,
                                    celerity::accessor::neighborhood<2>(1,1));
            auto new_dev = new_buf.get_access<cl::sycl::access::mode::
                                    discard_write>(cgh, celerity::accessor::one_to_one<2>());


            // as kernel name indicates, update matrix
            cgh.parallel_for<class Update>(
                cl::sycl::range<2>(N-2,N-2),
                cl::sycl::id<2>(1, 1),
                [=](cl::sycl::item<2> item) {
                    auto xp = item[0]+1;
                    auto yp = item[1]+1;
                    auto xm = item[0]-1;
                    auto ym = item[1]-1;

                    float theta_n = acosf(2.0f*(0.5f - myRand(item.get_linear_id())));
                    float phi_n = twoPi*myRand(item.get_linear_id());
                    float sen_theta = sinf(theta_n);

                    float new_x = sen_theta*cosf(phi_n);
                    float delta = new_x - mx_dev[item];

                    float E = mx_dev[{xm, item[1]}]*delta;
                    E += mx_dev[{xp, item[1]}]*delta;
                    E += mx_dev[{item[0], ym}]*delta;
                    E += mx_dev[{item[0], yp}]*delta;

                    if (E < 0.0f) {
                        new_dev[item] = new_x;
                    } else {
                        new_dev[item] = mx_dev[item];
                    }
                }
            );
        }
    );
}


int main (int argc, char* argv[]) {

    if (argc !=2 ) {
        std::cout << "Please enter a number to choose matriz size.\n";
        return EXIT_FAILURE;
    }

    const int N = atoi(argv[1]);

    // Fill Matrix with randoms
    std::vector<float> mx_host(N*N);
    auto seed = std::chrono::system_clock::now().time_since_epoch().count();
    auto gen = std::minstd_rand{seed};
    auto dist = std::uniform_real_distribution(0.0f, 1.0f);
    std::generate(mx_host.begin(), mx_host.end(),
             [&] {
                    int aux1 = acosf(2.0f*(0.5f- dist(gen));
                    int aux2 = twoPi*dist(gen);
                    return sinf(aux1)*cosf(aux2);
                 });
    

    celerity::buffer<float, 2> mx_buf(mx_host.data(), cl::sycl::range<2>(N, N));
    celerity::buffer<float, 2> new_buf(cl::sycl::range<2>(N, N));

    celerity::distr_queue queue;

    update(queue, mx_buf, new_buf, N);
    sides(queue, mx_buf, new_buf, N);
    corners(queue, mx_buf, new_buf, N);

    /*
    Here one would either take some measurents or write down to a file
    but since this is a test to see how simpler the code gets using Celerity
    instead of CUDA, it will left like this for the time being. 
    */

    std::ofstream last_state;
    last_state.open(std::string("last_state"));

    queue.submit(

        [=](celerity::handler& cgh) {

            auto new_dev = new_buf.get_access<cl::sycl::access::mode::read>(cgh,
                                                    celerity::accessor::all<2>());
            cgh.host_task(
                celerity::on_master_node, [=]() {

                    int rbx = 0;
                    for (int j=0; j<N; j++){
                        for (int i=0; i<N; ++i){
                            last_state << new_buf[{i,j}] << "   " ;
                        }
                        last_state << "\n";
                    }
                }
            );
        }
    );
    return EXIT_SUCCESS;
}