#include <vector>
#include <fstream>
#include <unordered_set>
#include <bitset>
#include <string>
#include <sparsehash/sparse_hash_set>
#include <boost/numpy.hpp>

namespace bp = boost::python;
namespace bn = boost::numpy;

namespace NSubsetGenerator{

using std::vector;
const size_t MAX_BITSET_SIZE = 192;
typedef std::bitset<MAX_BITSET_SIZE> bitset;
typedef vector<short> index_list;
typedef vector<vector<bool> > matrix;


template<typename T>
bn::ndarray vector_to_ndarray(const T &vector) {
    Py_intptr_t shape[1] = { vector.size() };
    bn::ndarray result = bn::zeros(
        1,
        shape,
        bn::dtype::get_builtin<typename T::value_type>()
    );
    std::copy(
        vector.begin(),
        vector.end(),
        reinterpret_cast<typename T::value_type*>(result.get_data())
    );
    return result;
}


matrix ndarray_to_matrix(const bn::ndarray& array) {
    bp::tuple shape = bp::extract<bp::tuple>(array.attr("shape"));
    size_t rows_count = bp::extract<size_t>(shape[0]);
    size_t columns_count = bp::extract<size_t>(shape[1]);
    matrix ans(rows_count);
    for (size_t row_index = 0; row_index < rows_count; ++row_index) {
        ans[row_index].resize(columns_count);
        bn::ndarray row = bp::extract<bn::ndarray>(array[row_index]);
        bool* data_ptr = reinterpret_cast<bool*>(row.get_data());
        std::copy(data_ptr, data_ptr + columns_count, ans[row_index].begin());
    }
    return ans;
}


index_list to_index_list(const bitset& bs) {
    index_list ans;
    for (size_t i = 0; i < bs.size(); ++i) {
        if (bs[i]) {
            ans.push_back(i);
        }
    }
    return ans;
}


bitset to_bitset(const index_list& ind_list) {
    bitset ans;
    for (int i = 0; i < ind_list.size(); ++i) {
        ans.set(ind_list[i]);
    }
    return ans;
}


template<typename T>
void clear_vector(vector<T>& to_clear) {
    vector<T> tmp;
    tmp.swap(to_clear);
}


class TSubsetGenerator{
    typedef google::sparse_hash_set<bitset, std::hash<bitset> > set;
    //typedef std::unordered_set<bitset> set;


    vector<bitset> sets;

    vector<bitset> get_simple_sets(const matrix& matr) const {
        vector<bitset> ans;

        size_t matr_size_x = matr.size();
        size_t matr_size_y = matr[0].size();

        for (size_t column = 0; column < matr_size_y; ++column) {
            index_list column_result;
            for (size_t row = 0; row < matr_size_x; ++row) {
                if (matr[row][column]) {
                    column_result.push_back(row);
                }
            }
            if (column_result.size() > 0) {
                ans.push_back(to_bitset(column_result));
            }
        }
        return ans;
    }

    void update(set& to_update, bitset element) const {
        bitset empty;

        set elements_to_add;
        elements_to_add.set_deleted_key(bitset());

        if (element != empty) {
            elements_to_add.insert(element);
        }

        for (auto it = to_update.begin(); it != to_update.end(); ++it) {
            bitset intersection = *it;
            intersection &= element;
            if (intersection != empty &&
                to_update.find(intersection) == to_update.end()) {
                elements_to_add.insert(intersection);
            }
        }

        while (elements_to_add.size() > 0) {
            auto element = *elements_to_add.begin();
            to_update.insert(element);
            elements_to_add.erase(element);
        }
    }

    void generate_and_set_matrix(const matrix& matr) {
        clear_vector(sets);

        vector<bitset> simple_sets = get_simple_sets(matr);

        set result_set;
        result_set.set_deleted_key(bitset());

        for (size_t i = 0; i < simple_sets.size(); ++i) {
            update(result_set, simple_sets[i]);
        }

        while (result_set.size() > 0) {
            auto element = *result_set.begin();
            sets.push_back(element);
            result_set.erase(element);
        }
    }

public:
    TSubsetGenerator() {
    }

    void generate_and_set(const bn::ndarray& input_matrix) {
        generate_and_set_matrix(ndarray_to_matrix(input_matrix));
    }

    void store(std::string filename) const {
        std::ofstream fout(filename);
        fout << sets.size() << '\n';
        for (auto it = sets.begin(); it != sets.end(); ++it) {
            index_list set_index_list = to_index_list(*it);
            fout << set_index_list.size() << ' ';
            for (size_t i = 0; i < set_index_list.size(); ++i) {
                fout << set_index_list[i] << ' ';
            }
            fout << '\n';
        }
        fout.flush();
        fout.close();
    }

    void load(std::string filename) {
        clear_vector(sets);
        std::ifstream fin;
        size_t sets_size;
        fin >> sets_size;
        sets.reserve(sets_size);
        for (size_t set_index = 0; set_index < sets_size; ++set_index) {
            size_t set_size;
            fin >> set_size;
            index_list set_index_list;
            for (size_t i = 0; i < set_size; ++i) {
                size_t element_index;
                fin >> element_index;
                set_index_list.push_back(element_index);
            }
            sets.push_back(to_bitset(set_index_list));
        }
    }

    size_t get_sets_count() const {
        return sets.size();
    }

    bn::ndarray get_set(size_t index) const {
        return vector_to_ndarray(to_index_list(sets[index]));
    }
};


} //namespace NSubsetGenerator


BOOST_PYTHON_MODULE(generate_subsets) {
    bn::initialize();
    bp::class_<NSubsetGenerator::TSubsetGenerator>("SubsetGenerator")
        .def("generate_and_set", &NSubsetGenerator::TSubsetGenerator::generate_and_set)
        .def("store", &NSubsetGenerator::TSubsetGenerator::store)
        .def("load", &NSubsetGenerator::TSubsetGenerator::load)
        .def("get_sets_count", &NSubsetGenerator::TSubsetGenerator::get_sets_count)
        .def("get_set", &NSubsetGenerator::TSubsetGenerator::get_set)
    ;
};
