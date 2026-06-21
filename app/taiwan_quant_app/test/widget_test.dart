// This is a basic Flutter widget test.
//
// To perform an interaction with a widget in your test, use the WidgetTester
// utility in the flutter_test package. For example, you can send tap and scroll
// gestures. You can also use WidgetTester to find child widgets in the widget
// tree, read text, and verify that the values of widget properties are correct.

import 'package:flutter/material.dart';
import 'package:flutter_test/flutter_test.dart';

import 'package:taiwan_quant_app/main.dart';

void main() {
  testWidgets('App shows stock scan UI', (WidgetTester tester) async {
    // Build our app and trigger a frame.
    await tester.pumpWidget(const QuantApp());

    // Verify the search input and scan button are present.
    expect(find.text('輸入股票代號 (例: 2330)'), findsOneWidget);
    expect(find.text('掃描'), findsOneWidget);
    expect(find.byIcon(Icons.search), findsOneWidget);
  });
}
