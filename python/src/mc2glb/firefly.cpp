#include <bits/stdc++.h>
using namespace std;
#define ll long long

void solve() {
    ll n, q;
    cin >> n >> q;
    vector<ll> a(n);
    for (int i = 0; i < n; i++) {
        cin >> a[i];
    }

    vector<ll> prefix(2 * n + 1);
    for (int i = 0; i < n; i++) {
        prefix[i + 1] = prefix[i] + a[i];
    }

    for (int i = n; i < 2 * n; i++) {
        prefix[i + 1] = prefix[i] + a[i - n];
    }

    for (int i = 0; i < q; i++) {
        int l, r;
        cin >> l >> r;
        
        /* left bit */
        // right bound of the left bit is going to be ceil(l / n)
        int right_l = (l + (n - 1))/n
    }
}

int main() {
    cin.tie(0)->sync_with_stdio(0);
    int t; cin >> t;
    while (t--) solve();
}