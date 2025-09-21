# Dataset Description

In this lab you will be attempting to detect fraudulent transactions using a synthetic dataset. The dataset includes 50k customers and 5k payment terminals. You will be given ~3M labeled transactions (fraud = 0|1) for training and ~71k unlabeled transactions for evaluation. The locations of the terminals and customers are within the same 100x100 cartesian grid.

DO NOT ATTEMPT TO READ ANY OF THE CSV FILES DIRECTLY, THEY ARE TOO LARGE AND WILL OVERFLOW YOUR CONTEXT WINDOW.
IF YOU NEED TO EXTRACT DATA FROM THE CSV FILES, WRITE CODE TO EXTRACT THE AGGREGATED DATA YOU NEED.

The following four datasets are provided in the `data/Payments Fraud DataSet/` directory.

## Data Set 1 - customers

50,000 customers along with their residence location on a 100x100 grid

- `CUSTOMER_ID` - STRING - Unique identifier of customer
  - 1000153411661301 ... 9999981251526109 (50000 unique values)
- `x_customer_id` - FLOAT - x-coordinate of customer in (0,100)
- `y_customer_id` - FLOAT - y-coordinate of customer in (0,100)

## Data Set 2 - terminals

5,000 terminals along with their locations on a 100x100 grid

- `TERMINAL_ID` - STRING - Unique identifier of terminal
  - 10024845 ... 99997987 (5000 unique values)
- `x_terminal_id` - FLOAT - x-coordinate of terminal in (0,100)
- `y_terminal_id` - FLOAT - y-coordinate of terminal in (0,100)

## Data Set 3 - merchants

~30k merchants records

- `MERCHANT_ID` - STRING- Unique ID of the merchant
- `BUSINESS_TYPE` - STRING- The business structure or type of the company e.g. Corporation, Limited Liability Company (LLC)
  - Corporations
  - Limited Liability Company (LLC)
  - S Corporations
  - Sole Proprietorships
- `MCC_CODE` - STRING - Four-digit numbers that describe a merchant's primary business activities. MCCs are used by credit card issuers, acquirers, schemes to identify the type of business in which a merchant is engaged.
  - 1520 ... 9950 (296 unique values)
- `LEGAL_NAME` - STRING - The name of the legal entity
  - 00002c ... ffff44 (30416 unique values)
- `FOUNDATION_DATE` - DATE - The foundation or incorporate date, the date when the company was founded
- `TAX_EXCEMPT_INDICATOR` - BOOLEAN- Indicates if the company is free from tax payment
  - False
  - True
- `OUTLET_TYPE` - STRING- Indicates the type of payments supported for the particular location e.g. Face to face, eCommerce or both
  - Ecommerce
  - Face to Face
  - Face to Face and Ecommerce
- `ACTIVE_FROM` - DATE- The date when the company become active
- `TRADING_FROM` - DATE- The date when the company started trading
- `ANNUAL_TURNOVER_CARD` - INTEGER- The estimated annual sales which were processed as card transactions
- `ANNUAL_TURNOVER` - INTEGER- The estimated annual total sales irrespective of the payment method
- `AVERAGE_TICKET_SALE_AMOUNT` - INTEGER- The average amount of a sale. It can vary depending on the type of business e.g. luxury goods, or car dealers might have higher average ticket sale amount.
- `PAYMENT_PERCENTAGE_FACE_TO_FACE` - INTEGER- The percentage of face to face payments from the total sales
- `PAYMENT_PERCENTAGE_ECOM` - INTEGER- The percentage of eCommerce payments from the total sales
- `PAYMENT_PERCENTAGE_MOTO` - INTEGER- The percentage of MOTO payments from the total sales. MOTO stands for Mail Order/Telephone Order (MOTO) transaction and it is a card-not-present transaction where the shopper provides their order and payment details by regular mail (not email), fax, or telephone.
- `DEPOSIT_REQUIRED_PERCENTAGE` - INTEGER- In general merchants are required to keep a reserve or a deposit with the payment provider to mitigate any potential risks associated with payment processing (e.g. chargebacks or any other risks)
- `The` percentage represents the required percentage of deposits compared to turnover
- `DEPOSIT_PERCENTAGE` - INTEGER- The actual deposit percentage from the turnover
- `DELIVERY_SAME_DAYS_PERCENTAGE` - INTEGER- The percentage of goods and services delivered in the same day by the merchant
- `DELIVERY_WEEK_ONE_PERCENTAGE` - INTEGER- The percentage of goods and services delivered by the merchant in the first week after the order was issued
- `DELIVERY_WEEK_TWO_PERCENTAGE` - INTEGER- The percentage of goods and services delivered by the merchant in week two after the order was issued
- `DELIVERY_OVER_TWO_WEEKS_PERCENTAGE` - INTEGER- The percentage of goods and services delivered by the merchant in after two weeks.

## Data Set 4 - transactions_train

~3M labeled transactions including the customer ID and terminal ID, time of the transaction, amount and fraud/not-fraud.

- `TX_ID` - Unique ID of transaction
- `TX_TS` - TIMESTAMP - Timestamp of transaction
- `CUSTOMER_ID` - STRING - Unique identifier of customer
- `TERMINAL_ID` - STRING - Unique identifier of terminal
- `TX_AMOUNT` - NUMERIC - Dollar amount of transaction
- `TX_FRAUD` - INT - LABEL - 1 if fraudulent, 0 if not.
- `TRANSACTION_GOODS_AND_SERVICES_AMOUNT` - NUMERIC - Amount of the transaction which accounts for goods and services.
- `TRANSACTION_CASHBACK_AMOUNT` - NUMERIC - Amount of the transaction which account for cashback.
- `CARD_EXPIRY_DATE` - STRING- The expiry date of the card in the MM/YY format.
  - 01/21 ... 12/23 (40 unique values)
- `CARD_DATA` - STRING- Masked PAN number of the card associated with the transaction
  - 3407********100 ... 6011********999 (14393 unique values)
- `CARD_BRAND` - STRING- The Brand of the card associated with the transaction (e.g. Visa, MasterCard)
  - AMEX
  - Discover
  - MasterCard
  - Visa
- `TRANSACTION_TYPE` - STRING- The type of payment of the current transaction (e.g. Purchase, Refund, Purchase with cashback)
  - Cash Advance/Withdrawal
  - Purchase
  - Purchase with cashback
  - Refund
- `TRANSACTION_STATUS` - STRING- The status of the current transaction. In general a payment transaction can go through multiple statuses, some technical and some with business relevance like Authorized indicating that the payment transaction has been authorized by the card issuer, or Settled indicating that the payment transaction has been settled with all the parties.
  - Authorized
  - Captured
  - Rejected
  - Settled
- `FAILURE_CODE` - STRING- In case of rejection the credit card failure code
  - 0 ... Z3 (27 unique values)
- `FAILURE_REASON` - STRING- In case of rejection the reason of the rejection
  - Approved or completed successfully
  - Cannot verify PIN
  - Capture card / Pick-up
  - Credit issuer unavailable
  - Do not honor
  - Error
  - Expired card
  - Insufficient funds/over credit limit / Not sufficient funds
  - Invalid amount
  - Invalid card number
  - Invalid merchant
  - Invalid transaction
  - Offline-declined
  - Pickup card, special condition
  - Refer to card issuer
  - Restricted card
  - Suspected fraud
  - Unable to go online; offline-declined
- `TRANSACTION_CURRENCY` - STRING- The currency of the current transaction
  - AED
  - CAD
  - CHF
  - CNY
  - EUR
  - GBP
  - HKD
  - JPY
  - MDL
  - RON
  - USD
- `CARD_COUNTRY_CODE` - STRING- The country code where the card was issued
  - AE
  - BE
  - BT
  - CA
  - DE
  - DK
  - FI
  - FR
  - GB
  - HR
  - NO
  - RO
  - SA
  - US
- `MERCHANT_ID` - STRING- The identifier of the merchant which is making the sale.
  - 0001dad0-0e30-490d-bf47-3c7e9cec4055 ... ffff7678-912c-4958-86f3-c9ebc862cbe1 (30452 unique values)
- `IS_RECURRING_TRANSACTION` - BOOLEAN- If the transaction is part of a subscription or installment plan the indicator will be Y
  - Fals
  - False
  - True
- `ACQUIRER_ID` - STRING- The acquirer which is processing the transaction. Depending on the setup the merchant can process payments via a PSP (payment service provider) which can potentially redirect the transactions to multiple acquirers.
  - ACQ1
  - ACQ2
  - ACQ3
  - ACQ4
  - ACQ5
  - ACQ6
- `CARDHOLDER_AUTH_METHOD` - STRING- The authentication method used by the cardholder
  - No CVM performed
  - Offline enciphered PIN
  - Offline enciphered PIN and signature
  - Offline plaintext PIN
  - Offline plaintext PIN and signature
  - Online PIN
  - Signature

## Data Set 5 - transactions_test

~71k unlabeled transactions

- `TX_ID` - Unique ID of transaction
- `TX_TS` - TIMESTAMP - Timestamp of transaction
- `CUSTOMER_ID` - STRING - Unique identifier of customer
- `TERMINAL_ID` - STRING - Unique identifier of terminal
- `TX_AMOUNT` - NUMERIC - Dollar amount of transaction
- `TRANSACTION_GOODS_AND_SERVICES_AMOUNT` - NUMERIC - Amount of the transaction which accounts for goods and services.
- `TRANSACTION_CASHBACK_AMOUNT` - NUMERIC - Amount of the transaction which account for cashback.
- `CARD_EXPIRY_DATE` - STRING- The expiry date of the card in the MM/YY format.
- `CARD_DATA` - STRING- Masked PAN number of the card associated with the transaction
- `CARD_BRAND` - STRING- The Brand of the card associated with the transaction (e.g. Visa, MasterCard)
- `TRANSACTION_TYPE` - STRING- The type of payment of the current transaction (e.g. Purchase, Refund, Purchase with cashback)
- `TRANSACTION_STATUS` - STRING- The status of the current transaction. In general a payment transaction can go through multiple statuses, some technical and some with business relevance like Authorized indicating that the payment transaction has been authorized by the card issuer, or Settled indicating that the payment transaction has been settled with all the parties.
- `FAILURE_CODE` - STRING- In case of rejection the credit card failure code
- `FAILURE_REASON` - STRING- In case of rejection the reason of the rejection
- `TRANSACTION_CURRENCY` - STRING- The currency of the current transaction
- `CARD_COUNTRY_CODE` - STRING- The country code where the card was issued
- `MERCHANT_ID` - STRING- The identifier of the merchant which is making the sale.
- `IS_RECURRING_TRANSACTION` - BOOLEAN- If the transaction is part of a subscription or installment plan the indicator will be Y
- `ACQUIRER_ID` - STRING- The acquirer which is processing the transaction. Depending on the setup the merchant can process payments via a PSP (payment service provider) which can potentially redirect the transactions to multiple acquirers.