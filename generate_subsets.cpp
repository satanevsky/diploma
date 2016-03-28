#include <vector>
#include <fstream>
#include <map>
#include <iostream>
#include <algorithm>
#include <set>
#include <assert.h>
#include <bitset>
#include <string>
#include <sparsehash/sparse_hash_set>


namespace NSubsetGenerator{

using std::vector;
const size_t MAX_BITSET_SIZE = 192;
typedef std::bitset<MAX_BITSET_SIZE> bitset;
typedef vector<short> index_list;
typedef vector<vector<bool> > matrix;

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

        elements_to_add.insert(element);

        for (auto it = to_update.begin(); it != to_update.end(); ++it) {
            bitset intersection = *it;
            intersection &= element;
            if (intersection != empty &&
                to_update.find(intersection) == to_update.end()) {
                elements_to_add.insert(intersection);
            }
        }

        while (elements_to_add.size() > 0) {
            to_update.insert(*elements_to_add.begin());
            elements_to_add.erase(elements_to_add.begin());
        }
    }

    void generate_and_set(const matrix& matr) {
        clear_vector(sets);

        vector<bitset> simple_sets = get_simple_sets(matr);

        set result_set;

        for (size_t i = 0; i < simple_sets.size(); ++i) {
            update(result_set, simple_sets[i]);
        }

        while (result_set.size() > 0) {
            sets.push_back(*result_set.begin());
            result_set.erase(result_set.begin());
        }
    }

public:
    TSubsetGenerator() {
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
};


} //namespace NSubsetGenerator

int main(int argc, char *argv[]) {
    NSubsetGenerator::TSubsetGenerator generator;
    return 0;
}
