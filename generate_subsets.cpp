#include <vector>
#include <fstream>
#include <map>
#include <iostream>
#include <algorithm>
#include <set>
#include <assert.h>
#include <bitset>
#include <unordered_set>
#include <unordered_map>
#include <string>
using namespace std;

vector<vector<short> > v1;
int threshold;

string outname;


void input() {
    ifstream in("input.txt");
    int n, m;
    in >> n >> m >> threshold;
    //threshold = 1;
    v1 = vector<vector<short> >(n, vector<short>(m, 0));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < m; ++j) {
            in >> v1[i][j];
        }
    }
}

vector<vector<short> > get_start() {
    vector<vector<short> > start;
    cout << v1.size() << ' ' << v1[0].size() << endl;
    for (int i = 0; i < v1[0].size(); ++i) {
        int cnt1 = 0, cnt0 = 0;
        for (int j = 0; j < v1.size(); ++j) {
            if (v1[j][i] == 1) ++cnt1;
	    if (v1[j][i] == 0) ++cnt0;
        }
	
	vector<short> cand;
	for (int j = 0; j < v1.size(); ++j) {
		if (cnt1 < cnt0) {
			if (v1[j][i] == 1) cand.push_back(j);
		} else {
			if (v1[j][i] == 0) cand.push_back(j);
		}
	}
	
       	if (cand.size() >= threshold) start.push_back(cand);
    }
    return start;
}


vector<short> my_merge(const vector<short> &a1, const vector<short> &a2) {
	vector<short> ans;
	int i = 0, j = 0;
	while (i < a1.size() && j < a2.size()) {
		while(i < a1.size() && j < a2.size() && a1[i] < a2[j]) ++i;
		while(i < a1.size() && j < a2.size() && a2[j] < a1[i]) ++j;
		if (i < a1.size() && j < a2.size() && a1[i] == a2[j]) {
			ans.push_back(a1[i]);
			++i;
			++j;
		}	
		
	}
	return ans;
}


int my_merge_size(const vector<short> &a1, const vector<short> &a2) {
	int ans = 0;
	int i = 0, j = 0;
	while (i < a1.size() && j < a2.size()) {
		while(i < a1.size() && j < a2.size() && a1[i] < a2[j]) ++i;
		while(i < a1.size() && j < a2.size() && a2[j] < a1[i]) ++j;
		if (i < a1.size() && j < a2.size() && a1[i] == a2[j]) {
			++ans;
			++i;
			++j;
		}	
		
	}
	return ans;
}

const int BITSET_SIZE = 150;
typedef bitset<BITSET_SIZE> bs;

vector<short> to_s(bs a) {
	vector<short> ans;
	for (int i = 0; i < a.size(); ++i) {
		if (a[i]) ans.push_back(i);
	}
	return ans;
}


bs to_bs(vector<short> a) {
	bs ans;
	for (int i = 0; i < a.size(); ++i) {
		ans.set(a[i]);
	}
	return ans;
}


class Comp {
public:
  bool operator()(const bs& x, const bs& y) const {
    for (int i = 0; i < BITSET_SIZE; ++i) {
     	if (x[i] ^ y[i]) return y[i];
    }
    return false;
  }
};

vector<bs> v;
vector<int> h(1, -1);
int in_hash = 0;
int hash_mask = 0;

std::hash<bs> hash_fn;

int get_h_val(const bs& a) {
	return hash_fn(a) & hash_mask;
}


void put_sim(int i) {
	int h_val = get_h_val(v[i]);
	while(h[h_val] != -1) {
		++h_val;
		if (h_val == h.size()) h_val = 0;
	}
	h[h_val] = i;
}


void rehash() {
    auto new_h = vector<int> (h.size() << 1, -1);
    new_h.swap(h);
    hash_mask = (hash_mask << 1) + 1;
    for (int i = 0; i < new_h.size(); ++i) {
	if (new_h[i] != -1) put_sim(new_h[i]);
    }
}


bool contains(const bs& a) {
	int h_val = get_h_val(a);
	while(h[h_val] != -1 && v[h[h_val]] != a) {
		++h_val;
		if (h_val == h.size()) h_val = 0;
	}
	return h[h_val] != -1;
}



void put(int i) {
	if (((in_hash + 1) << 1) >= h.size()) {
		rehash();
	}
	put_sim(i);
	++in_hash;
}


void add(const bs& a) {
	if (!contains(a)) {
		v.push_back(a);
		put(v.size() - 1);
	}
}


void gen() {
    ofstream fout(outname);
    vector<vector<short> > start = get_start();
    int reserve_size = 100000000;
    v.reserve(reserve_size);
    h.reserve(reserve_size);
    int prev_val = 1;
    for (int i = 0; i < start.size(); ++i) {
	cout << "iteration " << i <<  " of " << start.size() << " ";
	cout.flush();
	auto start_bs = to_bs(start[i]);
	int v_size = v.size();
	add(start_bs);
	for (int v_iter = 0; v_iter < v_size; ++v_iter) {
		auto ans = v[v_iter] & start_bs;
		if (ans.count() >= threshold) {
			add(ans);
			fout << ans.count() << ' ';
			for (int j = 0; j < ans.size(); ++j) {
				if (ans[j]) fout << j << ' ';
			}
			fout << '\n';		
		}
	}
	cout << v.size() << endl;	
    }
    fout << "end" << endl;
}


int main(int argc, char *argv[]) {
    outname = string(argv[1]);
    cout << "hello" << endl;
    input();
    cout << "input ok" << endl;
    gen();
    cout << "all ok" << endl;
    return 0;
}
