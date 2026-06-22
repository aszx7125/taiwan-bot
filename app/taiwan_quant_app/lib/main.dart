import 'package:flutter/material.dart';
import 'package:http/http.dart' as http;
import 'dart:convert';
import 'package:flutter/foundation.dart';

void main() {
  runApp(const QuantApp());
}

class QuantApp extends StatelessWidget {
  const QuantApp({super.key});

  @override
  Widget build(BuildContext context) {
    return MaterialApp(
      title: '台股量化終端',
      theme: ThemeData.dark().copyWith(
        scaffoldBackgroundColor: const Color(0xFF121212),
        appBarTheme: const AppBarTheme(backgroundColor: Color(0xFF1E1E1E)),
        bottomNavigationBarTheme: const BottomNavigationBarThemeData(
          backgroundColor: Color(0xFF1A1A1A),
          selectedItemColor: Colors.tealAccent,
          unselectedItemColor: Colors.grey,
        ),
      ),
      home: const MainDashboard(),
    );
  }
}

class MainDashboard extends StatefulWidget {
  const MainDashboard({super.key});

  @override
  State<MainDashboard> createState() => _MainDashboardState();
}

class _MainDashboardState extends State<MainDashboard> {
  int _currentIndex = 1;
  final List<Widget> _pages = [
    const MarketScreen(),
    const ScanScreen(),
    const SettingsScreen(),
  ];

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: _pages[_currentIndex],
      bottomNavigationBar: BottomNavigationBar(
        currentIndex: _currentIndex,
        onTap: (index) => setState(() => _currentIndex = index),
        items: const [
          BottomNavigationBarItem(icon: Icon(Icons.public), label: '主頁戰情'),
          BottomNavigationBarItem(icon: Icon(Icons.radar), label: '單股掃描'),
          BottomNavigationBarItem(icon: Icon(Icons.settings), label: '策略控制'),
        ],
      ),
    );
  }
}

// ==========================================
// 🌍 分頁 1：主頁戰情 (升級群組切換與價格排版)
// ==========================================
class MarketScreen extends StatefulWidget {
  const MarketScreen({super.key});

  @override
  State<MarketScreen> createState() => _MarketScreenState();
}

class _MarketScreenState extends State<MarketScreen> {
  List<dynamic> _top20List = [];
  List<dynamic> _watchlistQuotes = [];
  bool _isLoading = true;
  bool _isWatchlistLoading = false;

  // 🔥 升級：分類自選群組名單
  final Map<String, String> _watchlists = {
    "我的關注": "2330,0056,00878,2317,2454,2603",
    "權值三雄": "2330,2317,2454",
    "高股息ETF": "0056,00878,00713,00919,00929",
    "航運鋼鐵": "2603,2609,2615,2002",
  };
  
  // 預設選中的群組
  String _currentGroup = "我的關注";

  // 替換為你的 Render 網址
  String get baseUrl => 'https://你的專案名稱.onrender.com';

  @override
  void initState() {
    super.initState();
    _fetchAllMarketData();
  }

  // 抓取所有資料 (初始化用)
  Future<void> _fetchAllMarketData() async {
    setState(() => _isLoading = true);
    try {
      final top20res = await http.get(Uri.parse('$baseUrl/api/v1/market/top20'));
      await _fetchWatchlistQuotes(_watchlists[_currentGroup]!);
      
      if (top20res.statusCode == 200) {
        setState(() {
          _top20List = jsonDecode(top20res.body)['top20'];
          _isLoading = false;
        });
      }
    } catch (e) {
      setState(() => _isLoading = false);
    }
  }

  // 獨立抓取特定自選群組資料 (切換標籤時使用)
  Future<void> _fetchWatchlistQuotes(String tickers) async {
    setState(() => _isWatchlistLoading = true);
    try {
      final watchres = await http.get(Uri.parse('$baseUrl/api/v1/market/watchlist?tickers=$tickers'));
      if (watchres.statusCode == 200) {
        setState(() {
          _watchlistQuotes = jsonDecode(watchres.body)['watchlist'];
        });
      }
    } catch (e) {
      // 異常處理
    } finally {
      setState(() => _isWatchlistLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('🌍 市場戰情中心', style: TextStyle(fontWeight: FontWeight.bold))),
      body: _isLoading
          ? const Center(child: CircularProgressIndicator(color: Colors.tealAccent))
          : RefreshIndicator(
              color: Colors.tealAccent,
              onRefresh: _fetchAllMarketData,
              child: ListView(
                padding: const EdgeInsets.all(16),
                children: [
                  // 大盤綜合胜率
                  Container(
                    padding: const EdgeInsets.all(16),
                    decoration: BoxDecoration(
                      gradient: const LinearGradient(colors: [Color(0xFF1E3C72), Color(0xFF2A5298)]),
                      borderRadius: BorderRadius.circular(16),
                    ),
                    child: const Column(
                      children: [
                        Text('🧠 全市場 AI 綜合多空淨額', style: TextStyle(color: Colors.white70)),
                        SizedBox(height: 8),
                        Text('多頭優勢 62.4%', style: TextStyle(fontSize: 28, fontWeight: FontWeight.bold, color: Colors.greenAccent)),
                        SizedBox(height: 4),
                        Text('LGBM 與 LSTM 模型判定：市場整體震盪偏多', style: TextStyle(color: Colors.white, fontSize: 13)),
                      ],
                    ),
                  ),
                  const SizedBox(height: 24),

                  // 🔥 升級：自選群組切換標籤列
                  Row(
                    children: [
                      const Icon(Icons.star, color: Colors.amber, size: 20),
                      const SizedBox(width: 8),
                      const Text('自選即時行情', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.white)),
                    ],
                  ),
                  const SizedBox(height: 12),
                  // 群組標籤 (Chips)
                  SingleChildScrollView(
                    scrollDirection: Axis.horizontal,
                    child: Row(
                      children: _watchlists.keys.map((groupName) {
                        final isSelected = _currentGroup == groupName;
                        return Padding(
                          padding: const EdgeInsets.only(right: 8.0),
                          child: ChoiceChip(
                            label: Text(groupName, style: TextStyle(color: isSelected ? Colors.black : Colors.white)),
                            selected: isSelected,
                            selectedColor: Colors.tealAccent,
                            backgroundColor: const Color(0xFF2A2A2A),
                            onSelected: (selected) {
                              if (selected && _currentGroup != groupName) {
                                setState(() => _currentGroup = groupName);
                                _fetchWatchlistQuotes(_watchlists[groupName]!); // 切換時自動去撈新群組的報價
                              }
                            },
                          ),
                        );
                      }).toList(),
                    ),
                  ),
                  const SizedBox(height: 12),

                  // 自選股報價卡片區塊
                  SizedBox(
                    height: 110,
                    child: _isWatchlistLoading
                        ? const Center(child: CircularProgressIndicator(color: Colors.tealAccent))
                        : _watchlistQuotes.isEmpty
                            ? const Center(child: Text('無自選股資料', style: TextStyle(color: Colors.grey)))
                            : ListView.builder(
                                scrollDirection: Axis.horizontal,
                                itemCount: _watchlistQuotes.length,
                                itemBuilder: (context, index) {
                                  final item = _watchlistQuotes[index];
                                  
                                  // 🔥 升級：處理價格無限小數問題
                                  final num rawPrice = item['price'] ?? 0;
                                  // 限制最多顯示 2 位小數，並移除多餘的 .00
                                  final String displayPrice = rawPrice.toDouble().toStringAsFixed(2).replaceAll(RegExp(r"([.]*0+)(?!.*\d)$"), "");

                                  return Container(
                                    width: 140, // 稍微調小寬度
                                    margin: const EdgeInsets.only(right: 12),
                                    padding: const EdgeInsets.all(12),
                                    decoration: BoxDecoration(
                                      color: const Color(0xFF1E1E1E),
                                    borderRadius: BorderRadius.circular(12),
                                      border: Border.all(color: Colors.tealAccent.withOpacity(0.3)),
                                    ),
                                    child: Column(
                                      crossAxisAlignment: CrossAxisAlignment.start,
                                      children: [
                                        Text(item['name'], style: const TextStyle(fontWeight: FontWeight.bold, color: Colors.white, fontSize: 15), overflow: TextOverflow.ellipsis),
                                        Text(item['ticker'], style: const TextStyle(color: Colors.grey, fontSize: 11)),
                                        const Spacer(),
                                        Row(
                                          mainAxisAlignment: MainAxisAlignment.spaceBetween,
                                          children: [
                                            // 🔥 升級：加入 FittedBox 防止文字過長破版
                                            Expanded(
                                              child: FittedBox(
                                                alignment: Alignment.centerLeft,
                                                fit: BoxFit.scaleDown,
                                                child: Text('\$$displayPrice', style: const TextStyle(color: Colors.greenAccent, fontSize: 18, fontWeight: FontWeight.bold)),
                                              ),
                                            ),
                                            const SizedBox(width: 4),
                                            const Icon(Icons.bolt, color: Colors.amber, size: 16)
                                          ],
                                        ),
                                      ],
                                    ),
                                  );
                                },
                              ),
                  ),
                  const SizedBox(height: 24),

                  // TOP 20
                  const Text('🎯 AI 多頭推薦 TOP 20', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.white)),
                  const SizedBox(height: 12),
                  ..._top20List.asMap().entries.map((entry) {
                    final idx = entry.key + 1;
                    final stock = entry.value;
                    return Container(
                      margin: const EdgeInsets.only(bottom: 12),
                      decoration: BoxDecoration(color: const Color(0xFF1E1E1E), borderRadius: BorderRadius.circular(12)),
                      child: ListTile(
                        leading: CircleAvatar(backgroundColor: Colors.tealAccent.withOpacity(0.1), child: Text('$idx', style: const TextStyle(color: Colors.tealAccent, fontWeight: FontWeight.bold))),
                        title: Text('${stock['ticker']} ${stock['name']}', style: const TextStyle(fontWeight: FontWeight.bold, color: Colors.white)),
                        subtitle: Text('多頭勝率: ${stock['win_prob']}% | 現價: \$${stock['current_price']}', style: const TextStyle(color: Colors.grey, fontSize: 13)),
                        trailing: const Icon(Icons.trending_up, color: Color(0xFF00E676)),
                      ),
                    );
                  }).toList(),
                ],
              ),
            ),
    );
  }
}

// ==========================================
// ⚡ 分頁 2：單股診斷 
// ==========================================
class ScanScreen extends StatefulWidget {
  const ScanScreen({super.key});

  @override
  State<ScanScreen> createState() => _ScanScreenState();
}

class _ScanScreenState extends State<ScanScreen> {
  Map<String, dynamic>? _stockData;
  bool _isLoading = false;
  final TextEditingController _tickerController = TextEditingController(text: '2330');

  String get apiUrl {
    final ticker = _tickerController.text.trim();
    return 'https://你的專案名稱.onrender.com/api/v1/scan/$ticker';
  }

  Future<void> fetchAiData() async {
    if (_tickerController.text.isEmpty) return;
    setState(() { _isLoading = true; _stockData = null; });
    try {
      final response = await http.get(Uri.parse(apiUrl));
      if (response.statusCode == 200) {
        setState(() => _stockData = jsonDecode(response.body));
      }
    } catch (e) {
      // 異常處理
    } finally {
      setState(() => _isLoading = false);
    }
  }

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      appBar: AppBar(title: const Text('⚡ 深度掃描與策略分析', style: TextStyle(fontWeight: FontWeight.bold)), elevation: 0),
      body: Padding(
        padding: const EdgeInsets.all(16.0),
        child: Column(
          children: [
            Row(
              children: [
                Expanded(
                  child: TextField(
                    controller: _tickerController,
                    keyboardType: TextInputType.number,
                    decoration: InputDecoration(
                      hintText: '輸入代號 (例: 2330)',
                      filled: true,
                      fillColor: const Color(0xFF2A2A2A),
                      border: OutlineInputBorder(borderRadius: BorderRadius.circular(8), borderSide: BorderSide.none),
                      prefixIcon: const Icon(Icons.search, color: Colors.grey),
                    ),
                  ),
                ),
                const SizedBox(width: 12),
                ElevatedButton(
                  onPressed: _isLoading ? null : fetchAiData,
                  style: ElevatedButton.styleFrom(
                    backgroundColor: Colors.tealAccent.shade400,
                    foregroundColor: Colors.black,
                    padding: const EdgeInsets.symmetric(vertical: 16, horizontal: 24),
                    shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(8)),
                  ),
                  child: _isLoading
                      ? const SizedBox(width: 20, height: 20, child: CircularProgressIndicator(color: Colors.black, strokeWidth: 2))
                      : const Text('掃描', style: TextStyle(fontSize: 16, fontWeight: FontWeight.bold)),
                ),
              ],
            ),
            const SizedBox(height: 16),
            Expanded(child: _buildResultArea()),
          ],
        ),
      ),
    );
  }

  Widget _buildResultArea() {
    if (_isLoading) return const Center(child: CircularProgressIndicator(color: Colors.tealAccent));
    if (_stockData == null) return const Center(child: Text('輸入股票代號以獲取真實策略詳情', style: TextStyle(color: Colors.grey)));

    final ai = _stockData!['ai_analysis'];
    final strat = _stockData!['strategy_analysis'];
    final double bestLong = ai['best_long_prob'];
    final double bestShort = ai['best_short_prob'];
    
    Color cardColor = bestLong > bestShort ? const Color(0xFF00E676) : const Color(0xFFFF1744);

    return SingleChildScrollView(
      child: Column(
        crossAxisAlignment: CrossAxisAlignment.stretch,
        children: [
          Container(
            padding: const EdgeInsets.all(20),
            decoration: BoxDecoration(color: const Color(0xFF1E1E1E), borderRadius: BorderRadius.circular(16), border: Border.all(color: cardColor.withOpacity(0.5))),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Text('${_stockData!['ticker']} ${_stockData!['name']}', style: const TextStyle(fontSize: 24, fontWeight: FontWeight.bold, color: Colors.white)),
                const SizedBox(height: 16),
                Row(
                  children: [
                    Expanded(child: _buildProbBox('多頭勝率', bestLong, const Color(0xFF00E676))),
                    const SizedBox(width: 12),
                    Expanded(child: _buildProbBox('空頭勝率', bestShort, const Color(0xFFFF1744))),
                  ],
                ),
                const Divider(color: Colors.white24, height: 30),
                Row(
                  mainAxisAlignment: MainAxisAlignment.spaceAround,
                  children: [
                    _buildPriceColumn('現價', _stockData!['current_price'], Colors.white),
                    _buildPriceColumn('壓力 (Res)', _stockData!['res_level'], Colors.orangeAccent),
                    _buildPriceColumn('支撐 (Sup)', _stockData!['sup_level'], Colors.lightBlueAccent),
                  ],
                )
              ],
            ),
          ),
          const SizedBox(height: 20),

          const Text('🔍 核心指標與 SMC 策略分析', style: TextStyle(fontSize: 18, fontWeight: FontWeight.bold, color: Colors.white)),
          const SizedBox(height: 12),
          Container(
            padding: const EdgeInsets.all(16),
            decoration: BoxDecoration(color: const Color(0xFF1A1A1A), borderRadius: BorderRadius.circular(16)),
            child: Column(
              children: [
                _buildStrategyRow('相對強弱 (RS Index)', strat['rs_status'], Colors.greenAccent),
                const Divider(color: Colors.white12),
                _buildStrategyRow('波動率狀態 (Volatility)', strat['volatility_status'], Colors.orangeAccent),
                const Divider(color: Colors.white12),
                _buildStrategyRow('SMC 流動性掠奪 (Sweep)', strat['is_liquidity_sweep'] ? '⚠️ 觸發流動性掠奪' : '正常項目', strat['is_liquidity_sweep'] ? Colors.purpleAccent : Colors.grey),
                const Divider(color: Colors.white12),
                // 嚴格依循使用者對 POC 與流動性分開定義的要求
                _buildStrategyRow('SMC 控制點 (POC)', strat['is_poc_rejection'] ? '🟢 POC 支撐拒絕測試成功' : '區間未觸及', strat['is_poc_rejection'] ? Colors.lightBlueAccent : Colors.grey),
                const Divider(color: Colors.white12),
                _buildStrategyRow('原始形態判定 (Pattern)', strat['raw_pattern'], Colors.white70),
              ],
            ),
          ),
        ],
      ),
    );
  }

  Widget _buildProbBox(String label, double prob, Color color) {
    return Container(
      padding: const EdgeInsets.all(12),
      decoration: BoxDecoration(color: color.withOpacity(0.1), borderRadius: BorderRadius.circular(12)),
      child: Column(
        children: [
          Text(label, style: TextStyle(color: color, fontWeight: FontWeight.bold)),
          Text('${prob.toStringAsFixed(1)}%', style: TextStyle(fontSize: 24, fontWeight: FontWeight.bold, color: color)),
        ],
      ),
    );
  }

  Widget _buildPriceColumn(String label, dynamic price, Color color) {
    return Column(children: [Text(label, style: const TextStyle(color: Colors.grey, fontSize: 12)), const SizedBox(height: 4), Text(price.toString(), style: TextStyle(color: color, fontSize: 18, fontWeight: FontWeight.bold))]);
  }

  Widget _buildStrategyRow(String title, String status, Color statusColor) {
    return Padding(
      padding: const EdgeInsets.symmetric(vertical: 8.0),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(title, style: const TextStyle(color: Colors.grey, fontSize: 14)),
          Text(status, style: TextStyle(color: statusColor, fontWeight: FontWeight.bold, fontSize: 14)),
        ],
      ),
    );
  }
}

class SettingsScreen extends StatelessWidget {
  const SettingsScreen({super.key});
  @override
  Widget build(BuildContext context) {
    return Scaffold(appBar: AppBar(title: const Text('⚙️ 策略控制')), body: const Center(child: Text('設定維護中...')));
  }
}