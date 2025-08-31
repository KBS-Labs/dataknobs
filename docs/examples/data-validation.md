# Data Validation Examples

This document provides comprehensive examples of data validation scenarios using the DataKnobs validation system, from simple field validation to complex business rules.

## Example 1: User Registration System

### Scenario
Complete user registration with multi-level validation, duplicate checking, and security requirements.

```python
from dataknobs_data.validation import (
    Schema, Required, Length, Pattern, Enum, Unique, Custom, Range
)
from dataknobs_data import Record, MemoryDatabase, Query
import re
import hashlib
from datetime import datetime, date
from typing import List, Dict, Any

class UserRegistrationSystem:
    """Complete user registration with comprehensive validation"""
    
    def __init__(self):
        self.user_db = MemoryDatabase()
        self.setup_schemas()
        self.setup_validators()
    
    def setup_schemas(self):
        """Define validation schemas for different user types"""
        
        # Base user schema
        self.base_user_schema = (Schema("BaseUser", strict=False)
            .field("username", "STRING", required=True,
                   constraints=[
                       Length(min=3, max=30),
                       Pattern(r"^[a-zA-Z0-9_-]+$", 
                              "Username can only contain letters, numbers, hyphens, and underscores"),
                       Unique("username"),
                       Custom(self.check_reserved_usernames, "Username is reserved")
                   ])
            .field("email", "STRING", required=True,
                   constraints=[
                       Pattern(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$",
                              "Invalid email format"),
                       Unique("email"),
                       Custom(self.check_disposable_email, "Disposable email addresses not allowed")
                   ])
            .field("password", "STRING", required=True,
                   constraints=[
                       Length(min=12, max=128),
                       Custom(self.validate_password_strength, "Password does not meet security requirements")
                   ])
            .field("age", "INTEGER", required=True,
                   constraints=[
                       Range(min=13, max=150, message="Age must be between 13 and 150")
                   ])
            .field("country", "STRING", required=True,
                   constraints=[
                       Custom(self.validate_country_code, "Invalid country code")
                   ])
            .field("terms_accepted", "BOOLEAN", required=True,
                   constraints=[
                       Custom(lambda x: x is True, "Terms and conditions must be accepted")
                   ])
            .field("marketing_consent", "BOOLEAN", default=False)
        )
        
        # Premium user schema (extends base)
        self.premium_user_schema = (Schema("PremiumUser", strict=False)
            .copy_from(self.base_user_schema)
            .field("payment_method", "DICT", required=True,
                   constraints=[
                       Custom(self.validate_payment_method, "Invalid payment method")
                   ])
            .field("billing_address", "DICT", required=True,
                   constraints=[
                       Custom(self.validate_address, "Invalid billing address")
                   ])
            .field("subscription_tier", "STRING", required=True,
                   constraints=[
                       Enum(["basic", "professional", "enterprise"])
                   ])
        )
        
        # Business user schema
        self.business_user_schema = (Schema("BusinessUser", strict=False)
            .copy_from(self.base_user_schema)
            .field("company_name", "STRING", required=True,
                   constraints=[
                       Length(min=2, max=100)
                   ])
            .field("vat_number", "STRING", required=False,
                   constraints=[
                       Custom(self.validate_vat_number, "Invalid VAT number")
                   ])
            .field("employees", "INTEGER", required=True,
                   constraints=[
                       Range(min=1, max=1000000)
                   ])
        )
    
    def setup_validators(self):
        """Setup custom validators"""
        
        # Reserved usernames
        self.reserved_usernames = {
            "admin", "root", "administrator", "moderator", "support",
            "help", "api", "www", "mail", "email", "blog", "news",
            "about", "contact", "privacy", "terms", "legal"
        }
        
        # Disposable email domains
        self.disposable_domains = {
            "tempmail.com", "throwaway.email", "guerrillamail.com",
            "mailinator.com", "10minutemail.com", "trashmail.com"
        }
        
        # Valid country codes (ISO 3166-1 alpha-2)
        self.valid_countries = {
            "US", "GB", "CA", "AU", "DE", "FR", "JP", "CN", "IN", "BR",
            # ... add all country codes
        }
    
    def check_reserved_usernames(self, username: str) -> bool:
        """Check if username is not reserved"""
        return username.lower() not in self.reserved_usernames
    
    def check_disposable_email(self, email: str) -> bool:
        """Check if email is not from disposable service"""
        domain = email.split("@")[1] if "@" in email else ""
        return domain.lower() not in self.disposable_domains
    
    def validate_password_strength(self, password: str) -> bool:
        """Validate password meets security requirements"""
        if len(password) < 12:
            return False
        
        # Check complexity requirements
        has_upper = re.search(r"[A-Z]", password) is not None
        has_lower = re.search(r"[a-z]", password) is not None
        has_digit = re.search(r"\d", password) is not None
        has_special = re.search(r"[!@#$%^&*(),.?\":{}|<>]", password) is not None
        
        # Require at least 3 out of 4 character types
        complexity_score = sum([has_upper, has_lower, has_digit, has_special])
        if complexity_score < 3:
            return False
        
        # Check for common patterns
        common_patterns = [
            r"12345", r"qwerty", r"password", r"admin",
            r"([a-z])\1{2,}",  # Repeated characters
            r"(012|123|234|345|456|567|678|789)",  # Sequential numbers
            r"(abc|bcd|cde|def|efg|fgh)",  # Sequential letters
        ]
        
        for pattern in common_patterns:
            if re.search(pattern, password.lower()):
                return False
        
        return True
    
    def validate_country_code(self, country: str) -> bool:
        """Validate ISO country code"""
        return country.upper() in self.valid_countries
    
    def validate_payment_method(self, payment: Dict[str, Any]) -> bool:
        """Validate payment method structure"""
        if not isinstance(payment, dict):
            return False
        
        payment_type = payment.get("type")
        
        if payment_type == "credit_card":
            return all([
                self.validate_credit_card(payment.get("card_number", "")),
                payment.get("expiry_month") in range(1, 13),
                payment.get("expiry_year") >= datetime.now().year,
                len(str(payment.get("cvv", ""))) in [3, 4]
            ])
        elif payment_type == "paypal":
            return self.validate_email(payment.get("paypal_email", ""))
        elif payment_type == "bank_transfer":
            return all([
                payment.get("account_number"),
                payment.get("routing_number"),
                payment.get("account_holder_name")
            ])
        
        return False
    
    def validate_credit_card(self, card_number: str) -> bool:
        """Validate credit card using Luhn algorithm"""
        card_number = re.sub(r'\D', '', str(card_number))
        
        if not card_number or len(card_number) < 13 or len(card_number) > 19:
            return False
        
        # Luhn algorithm
        total = 0
        reverse = card_number[::-1]
        for i, digit in enumerate(reverse):
            n = int(digit)
            if i % 2 == 1:
                n *= 2
                if n > 9:
                    n -= 9
            total += n
        
        return total % 10 == 0
    
    def validate_email(self, email: str) -> bool:
        """Extended email validation"""
        pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
        return re.match(pattern, email) is not None
    
    def validate_address(self, address: Dict[str, Any]) -> bool:
        """Validate address structure"""
        required_fields = ["street", "city", "country", "postal_code"]
        
        if not all(field in address for field in required_fields):
            return False
        
        # Validate postal code format based on country
        country = address.get("country")
        postal_code = address.get("postal_code")
        
        postal_patterns = {
            "US": r"^\d{5}(-\d{4})?$",  # ZIP code
            "GB": r"^[A-Z]{1,2}\d{1,2}[A-Z]?\s?\d[A-Z]{2}$",  # UK postcode
            "CA": r"^[A-Z]\d[A-Z]\s?\d[A-Z]\d$",  # Canadian postal code
            # Add more patterns
        }
        
        if country in postal_patterns:
            if not re.match(postal_patterns[country], postal_code):
                return False
        
        return True
    
    def validate_vat_number(self, vat_number: str) -> bool:
        """Validate EU VAT number format"""
        # Simplified VAT validation (real implementation would check with VIES)
        vat_patterns = {
            "GB": r"^GB\d{9}$",
            "DE": r"^DE\d{9}$",
            "FR": r"^FR[A-Z0-9]{11}$",
            # Add more EU country patterns
        }
        
        for country, pattern in vat_patterns.items():
            if re.match(pattern, vat_number):
                return True
        
        return False
    
    def register_user(self, user_data: Dict[str, Any], user_type: str = "base") -> Dict[str, Any]:
        """Register a new user with validation"""
        
        # Select appropriate schema
        if user_type == "premium":
            schema = self.premium_user_schema
        elif user_type == "business":
            schema = self.business_user_schema
        else:
            schema = self.base_user_schema
        
        # Create record
        user_record = Record(data=user_data)
        
        # Validate
        result = schema.validate(user_record, coerce=True)
        
        if not result.valid:
            return {
                "success": False,
                "errors": result.errors,
                "warnings": result.warnings
            }
        
        # Additional security checks
        security_check = self.perform_security_checks(result.value.data)
        if not security_check["passed"]:
            return {
                "success": False,
                "errors": security_check["errors"]
            }
        
        # Hash password before storage
        validated_data = result.value.data.copy()
        validated_data["password"] = self.hash_password(validated_data["password"])
        
        # Add metadata
        validated_data["created_at"] = datetime.now().isoformat()
        validated_data["status"] = "active"
        validated_data["verification_token"] = self.generate_verification_token(validated_data["email"])
        
        # Save to database
        self.user_db.insert(Record(data=validated_data))
        
        return {
            "success": True,
            "user_id": validated_data.get("username"),
            "verification_required": True,
            "message": "Registration successful. Please check your email for verification."
        }
    
    def perform_security_checks(self, user_data: Dict[str, Any]) -> Dict[str, Any]:
        """Additional security validation"""
        errors = []
        
        # Check password not similar to username/email
        password = user_data.get("password", "").lower()
        username = user_data.get("username", "").lower()
        email = user_data.get("email", "").split("@")[0].lower()
        
        if username in password or email in password:
            errors.append("Password too similar to username or email")
        
        # Check for SQL injection attempts
        suspicious_patterns = [
            r"(\b(SELECT|INSERT|UPDATE|DELETE|DROP|UNION|ALTER)\b)",
            r"(;|--|\||&&)",
            r"(<script|javascript:|onerror=)",
        ]
        
        for field, value in user_data.items():
            if isinstance(value, str):
                for pattern in suspicious_patterns:
                    if re.search(pattern, value, re.IGNORECASE):
                        errors.append(f"Suspicious content in {field}")
                        break
        
        return {
            "passed": len(errors) == 0,
            "errors": errors
        }
    
    def hash_password(self, password: str) -> str:
        """Hash password using secure algorithm"""
        # In production, use bcrypt or argon2
        salt = "secure_salt_here"  # Should be random per user
        return hashlib.pbkdf2_hmac('sha256', password.encode(), salt.encode(), 100000).hex()
    
    def generate_verification_token(self, email: str) -> str:
        """Generate email verification token"""
        timestamp = datetime.now().timestamp()
        token_data = f"{email}:{timestamp}"
        return hashlib.sha256(token_data.encode()).hexdigest()

# Usage example
def test_registration():
    system = UserRegistrationSystem()
    
    # Test valid registration
    valid_user = {
        "username": "john_doe_123",
        "email": "john@example.com",
        "password": "Secure#Pass123!@#",
        "age": 25,
        "country": "US",
        "terms_accepted": True,
        "marketing_consent": False
    }
    
    result = system.register_user(valid_user, "base")
    print(f"Registration result: {result}")
    
    # Test invalid registration
    invalid_user = {
        "username": "admin",  # Reserved
        "email": "test@tempmail.com",  # Disposable
        "password": "weak",  # Too weak
        "age": 10,  # Too young
        "country": "XX",  # Invalid country
        "terms_accepted": False  # Not accepted
    }
    
    result = system.register_user(invalid_user, "base")
    print(f"Invalid registration errors: {result.get('errors')}")

if __name__ == "__main__":
    test_registration()
```

## Example 2: Financial Transaction Validation

### Scenario
Validate financial transactions with compliance checks, fraud detection, and regulatory requirements.

```python
from decimal import Decimal
from dataknobs_data.validation import Schema, Range, Custom, Required, Pattern
from dataknobs_data import Record
import re
from datetime import datetime, timedelta
from typing import Dict, List, Optional

class FinancialTransactionValidator:
    """Comprehensive financial transaction validation"""
    
    def __init__(self):
        self.setup_schemas()
        self.transaction_history = []  # For pattern detection
        self.blacklisted_accounts = set()
        self.suspicious_patterns = []
    
    def setup_schemas(self):
        """Define validation schemas for different transaction types"""
        
        # Base transaction schema
        self.base_transaction_schema = (Schema("BaseTransaction", strict=True)
            .field("transaction_id", "STRING", required=True,
                   constraints=[
                       Pattern(r"^TXN-[0-9]{4}-[0-9]{6}-[A-Z0-9]{6}$",
                              "Invalid transaction ID format")
                   ])
            .field("amount", "FLOAT", required=True,
                   constraints=[
                       Range(min=0.01, max=1000000,
                            message="Amount must be between $0.01 and $1,000,000"),
                       Custom(self.validate_decimal_places, "Amount must have at most 2 decimal places")
                   ])
            .field("currency", "STRING", required=True,
                   constraints=[
                       Pattern(r"^[A-Z]{3}$", "Currency must be 3-letter ISO code")
                   ])
            .field("timestamp", "STRING", required=True,
                   constraints=[
                       Custom(self.validate_timestamp, "Invalid or future timestamp")
                   ])
            .field("account_from", "STRING", required=True,
                   constraints=[
                       Custom(self.validate_account_number, "Invalid account number"),
                       Custom(self.check_blacklist, "Account is blacklisted")
                   ])
            .field("account_to", "STRING", required=True,
                   constraints=[
                       Custom(self.validate_account_number, "Invalid account number"),
                       Custom(self.check_blacklist, "Account is blacklisted")
                   ])
        )
        
        # Wire transfer schema (with additional requirements)
        self.wire_transfer_schema = (Schema("WireTransfer", strict=True)
            .copy_from(self.base_transaction_schema)
            .field("swift_code", "STRING", required=True,
                   constraints=[
                       Pattern(r"^[A-Z]{6}[A-Z0-9]{2}([A-Z0-9]{3})?$",
                              "Invalid SWIFT/BIC code")
                   ])
            .field("reference", "STRING", required=True,
                   constraints=[
                       Length(min=1, max=140),
                       Custom(self.validate_reference, "Invalid reference format")
                   ])
            .field("beneficiary_name", "STRING", required=True,
                   constraints=[
                       Length(min=2, max=70),
                       Pattern(r"^[a-zA-Z\s\-'\.]+$", "Invalid beneficiary name")
                   ])
            .field("beneficiary_address", "DICT", required=True,
                   constraints=[
                       Custom(self.validate_wire_address, "Invalid beneficiary address")
                   ])
        )
        
        # ACH transfer schema
        self.ach_transfer_schema = (Schema("ACHTransfer", strict=True)
            .copy_from(self.base_transaction_schema)
            .field("routing_number", "STRING", required=True,
                   constraints=[
                       Pattern(r"^\d{9}$", "Routing number must be 9 digits"),
                       Custom(self.validate_routing_number, "Invalid routing number")
                   ])
            .field("sec_code", "STRING", required=True,
                   constraints=[
                       Enum(["PPD", "CCD", "WEB", "TEL", "ARC", "BOC", "POP"])
                   ])
            .field("effective_date", "STRING", required=True,
                   constraints=[
                       Custom(self.validate_effective_date, "Invalid effective date")
                   ])
        )
    
    def validate_decimal_places(self, amount: float) -> bool:
        """Validate amount has at most 2 decimal places"""
        decimal_amount = Decimal(str(amount))
        return decimal_amount.as_tuple().exponent >= -2
    
    def validate_timestamp(self, timestamp: str) -> bool:
        """Validate timestamp is valid and not in future"""
        try:
            dt = datetime.fromisoformat(timestamp)
            return dt <= datetime.now()
        except:
            return False
    
    def validate_account_number(self, account: str) -> bool:
        """Validate account number format"""
        # IBAN validation
        if account.startswith(("GB", "DE", "FR", "IT", "ES")):
            return self.validate_iban(account)
        # US account validation
        elif re.match(r"^\d{10,17}$", account):
            return True
        return False
    
    def validate_iban(self, iban: str) -> bool:
        """Validate IBAN using check digits"""
        iban = iban.replace(" ", "").upper()
        
        if not re.match(r"^[A-Z]{2}\d{2}[A-Z0-9]+$", iban):
            return False
        
        # Move first 4 chars to end
        rearranged = iban[4:] + iban[:4]
        
        # Replace letters with numbers (A=10, B=11, ..., Z=35)
        numeric = ""
        for char in rearranged:
            if char.isdigit():
                numeric += char
            else:
                numeric += str(ord(char) - ord('A') + 10)
        
        # Check modulo 97
        return int(numeric) % 97 == 1
    
    def check_blacklist(self, account: str) -> bool:
        """Check if account is not blacklisted"""
        return account not in self.blacklisted_accounts
    
    def validate_routing_number(self, routing: str) -> bool:
        """Validate US routing number with checksum"""
        if not re.match(r"^\d{9}$", routing):
            return False
        
        # ABA routing number checksum
        weights = [3, 7, 1, 3, 7, 1, 3, 7, 1]
        total = sum(int(routing[i]) * weights[i] for i in range(9))
        return total % 10 == 0
    
    def validate_reference(self, reference: str) -> bool:
        """Validate payment reference"""
        # Check for suspicious patterns
        suspicious = [
            r"(terrorist|terrorism|drugs|laundering)",
            r"(^|[^a-z])(iran|north korea|syria)([^a-z]|$)",
        ]
        
        for pattern in suspicious:
            if re.search(pattern, reference.lower()):
                return False
        
        return True
    
    def validate_wire_address(self, address: Dict) -> bool:
        """Validate wire transfer beneficiary address"""
        required = ["street", "city", "country"]
        return all(field in address and address[field] for field in required)
    
    def validate_effective_date(self, date_str: str) -> bool:
        """Validate ACH effective date (business days only)"""
        try:
            date = datetime.strptime(date_str, "%Y-%m-%d").date()
            # Check not weekend
            if date.weekday() >= 5:  # Saturday = 5, Sunday = 6
                return False
            # Check not too far in future (60 days max)
            if date > (datetime.now().date() + timedelta(days=60)):
                return False
            return True
        except:
            return False
    
    def perform_aml_checks(self, transaction: Dict) -> Dict[str, Any]:
        """Anti-Money Laundering checks"""
        risk_score = 0
        flags = []
        
        amount = Decimal(str(transaction.get("amount", 0)))
        
        # Check for structuring (multiple transactions just under reporting threshold)
        if 9000 < amount < 10000:
            risk_score += 20
            flags.append("Possible structuring")
        
        # Check for round amounts (common in money laundering)
        if amount % 1000 == 0 and amount > 5000:
            risk_score += 10
            flags.append("Round amount transaction")
        
        # Check transaction velocity
        velocity_risk = self.check_velocity(transaction)
        risk_score += velocity_risk["score"]
        if velocity_risk["flag"]:
            flags.append(velocity_risk["flag"])
        
        # Check for suspicious patterns
        pattern_risk = self.check_patterns(transaction)
        risk_score += pattern_risk["score"]
        flags.extend(pattern_risk["flags"])
        
        # Determine action based on risk score
        if risk_score >= 70:
            action = "block"
        elif risk_score >= 40:
            action = "review"
        else:
            action = "approve"
        
        return {
            "risk_score": risk_score,
            "flags": flags,
            "action": action,
            "requires_sar": risk_score >= 60  # Suspicious Activity Report
        }
    
    def check_velocity(self, transaction: Dict) -> Dict:
        """Check transaction velocity for anomalies"""
        account = transaction.get("account_from")
        amount = Decimal(str(transaction.get("amount", 0)))
        
        # Get recent transactions for this account
        recent = [t for t in self.transaction_history[-100:] 
                 if t.get("account_from") == account]
        
        if len(recent) >= 5:
            # Check if unusual number of transactions
            if len(recent) > 20:
                return {"score": 30, "flag": "High transaction velocity"}
            
            # Check if unusual amount
            amounts = [Decimal(str(t.get("amount", 0))) for t in recent]
            avg_amount = sum(amounts) / len(amounts)
            
            if amount > avg_amount * 5:
                return {"score": 25, "flag": "Amount significantly above average"}
        
        return {"score": 0, "flag": None}
    
    def check_patterns(self, transaction: Dict) -> Dict:
        """Check for known suspicious patterns"""
        flags = []
        score = 0
        
        # Rapid movement pattern (in -> out quickly)
        if self.detect_rapid_movement(transaction):
            flags.append("Rapid fund movement detected")
            score += 35
        
        # Circular transaction pattern
        if self.detect_circular_pattern(transaction):
            flags.append("Circular transaction pattern")
            score += 40
        
        return {"score": score, "flags": flags}
    
    def detect_rapid_movement(self, transaction: Dict) -> bool:
        """Detect rapid in-out fund movement"""
        account = transaction.get("account_to")
        
        # Check if this account received funds recently and is now sending
        for hist_txn in self.transaction_history[-20:]:
            if (hist_txn.get("account_to") == account and
                transaction.get("account_from") == account):
                # Funds received and sent within short time
                return True
        
        return False
    
    def detect_circular_pattern(self, transaction: Dict) -> bool:
        """Detect circular transaction patterns"""
        # Simplified check - in production would use graph analysis
        account_from = transaction.get("account_from")
        account_to = transaction.get("account_to")
        
        # Check if accounts have transacted in reverse recently
        for hist_txn in self.transaction_history[-50:]:
            if (hist_txn.get("account_from") == account_to and
                hist_txn.get("account_to") == account_from):
                return True
        
        return False
    
    def validate_transaction(self, transaction_data: Dict, 
                           transaction_type: str = "base") -> Dict[str, Any]:
        """Validate a financial transaction"""
        
        # Select appropriate schema
        if transaction_type == "wire":
            schema = self.wire_transfer_schema
        elif transaction_type == "ach":
            schema = self.ach_transfer_schema
        else:
            schema = self.base_transaction_schema
        
        # Create record
        transaction_record = Record(data=transaction_data)
        
        # Schema validation
        result = schema.validate(transaction_record, coerce=True)
        
        if not result.valid:
            return {
                "valid": False,
                "errors": result.errors,
                "transaction_id": transaction_data.get("transaction_id")
            }
        
        # AML checks
        aml_result = self.perform_aml_checks(result.value.data)
        
        # Sanctions screening (simplified)
        sanctions_result = self.screen_sanctions(result.value.data)
        
        # Add to history for pattern detection
        self.transaction_history.append(result.value.data)
        
        # Determine final status
        if aml_result["action"] == "block" or sanctions_result["blocked"]:
            status = "blocked"
            reason = aml_result["flags"] + sanctions_result.get("matches", [])
        elif aml_result["action"] == "review":
            status = "pending_review"
            reason = aml_result["flags"]
        else:
            status = "approved"
            reason = []
        
        return {
            "valid": True,
            "status": status,
            "transaction_id": transaction_data.get("transaction_id"),
            "risk_score": aml_result["risk_score"],
            "flags": reason,
            "requires_sar": aml_result["requires_sar"],
            "sanctions_hit": sanctions_result["blocked"]
        }
    
    def screen_sanctions(self, transaction: Dict) -> Dict:
        """Screen against sanctions lists"""
        # Simplified sanctions screening
        sanctioned_countries = {"Iran", "North Korea", "Syria", "Cuba"}
        sanctioned_entities = {"Bad Company LLC", "Evil Corp"}
        
        beneficiary = transaction.get("beneficiary_name", "")
        
        # Check country
        if "beneficiary_address" in transaction:
            country = transaction["beneficiary_address"].get("country", "")
            if country in sanctioned_countries:
                return {"blocked": True, "matches": [f"Sanctioned country: {country}"]}
        
        # Check entity
        if beneficiary in sanctioned_entities:
            return {"blocked": True, "matches": [f"Sanctioned entity: {beneficiary}"]}
        
        return {"blocked": False, "matches": []}

# Usage example
def test_financial_validation():
    validator = FinancialTransactionValidator()
    
    # Valid wire transfer
    valid_wire = {
        "transaction_id": "TXN-2024-000001-ABC123",
        "amount": 50000.00,
        "currency": "USD",
        "timestamp": datetime.now().isoformat(),
        "account_from": "GB82WEST12345698765432",
        "account_to": "DE89370400440532013000",
        "swift_code": "DEUTDEFF",
        "reference": "Invoice payment Q1-2024",
        "beneficiary_name": "Acme Corporation",
        "beneficiary_address": {
            "street": "123 Business St",
            "city": "Frankfurt",
            "country": "Germany"
        }
    }
    
    result = validator.validate_transaction(valid_wire, "wire")
    print(f"Wire transfer validation: {result}")
    
    # Suspicious transaction
    suspicious = {
        "transaction_id": "TXN-2024-000002-XYZ789",
        "amount": 9999.99,  # Just under reporting threshold
        "currency": "USD",
        "timestamp": datetime.now().isoformat(),
        "account_from": "1234567890",
        "account_to": "0987654321",
    }
    
    result = validator.validate_transaction(suspicious, "base")
    print(f"Suspicious transaction: {result}")

if __name__ == "__main__":
    test_financial_validation()
```

## Example 3: Healthcare Data Quality Validation

### Scenario
Validate healthcare records for data quality, HIPAA compliance, and clinical accuracy.

```python
from dataknobs_data.validation import Schema, Custom, Pattern, Range, Required
from dataknobs_data import Record
import re
from datetime import datetime, date
from typing import Dict, List, Optional

class HealthcareDataValidator:
    """Healthcare data validation with HIPAA compliance"""
    
    def __init__(self):
        self.setup_schemas()
        self.setup_medical_references()
    
    def setup_schemas(self):
        """Define healthcare record schemas"""
        
        # Patient demographics schema
        self.patient_schema = (Schema("PatientRecord", strict=False)
            .field("patient_id", "STRING", required=True,
                   constraints=[
                       Pattern(r"^PAT-\d{10}$", "Invalid patient ID format"),
                       Custom(self.validate_not_ssn, "Patient ID cannot be SSN")
                   ])
            .field("mrn", "STRING", required=True,
                   constraints=[
                       Pattern(r"^MRN-\d{8}$", "Invalid MRN format")
                   ])
            .field("name", "DICT", required=True,
                   constraints=[
                       Custom(self.validate_name_structure, "Invalid name structure")
                   ])
            .field("dob", "STRING", required=True,
                   constraints=[
                       Custom(self.validate_dob, "Invalid date of birth")
                   ])
            .field("gender", "STRING", required=True,
                   constraints=[
                       Enum(["M", "F", "O", "U"])  # Male, Female, Other, Unknown
                   ])
            .field("contact", "DICT", required=True,
                   constraints=[
                       Custom(self.validate_contact_info, "Invalid contact information")
                   ])
        )
        
        # Clinical data schema
        self.clinical_schema = (Schema("ClinicalRecord", strict=True)
            .field("patient_id", "STRING", required=True)
            .field("encounter_id", "STRING", required=True,
                   constraints=[
                       Pattern(r"^ENC-\d{12}$", "Invalid encounter ID")
                   ])
            .field("encounter_date", "STRING", required=True,
                   constraints=[
                       Custom(self.validate_encounter_date, "Invalid encounter date")
                   ])
            .field("vital_signs", "DICT", required=False,
                   constraints=[
                       Custom(self.validate_vital_signs, "Invalid vital signs")
                   ])
            .field("diagnoses", "LIST", required=True,
                   constraints=[
                       Custom(self.validate_diagnoses, "Invalid diagnosis codes")
                   ])
            .field("procedures", "LIST", required=False,
                   constraints=[
                       Custom(self.validate_procedures, "Invalid procedure codes")
                   ])
            .field("medications", "LIST", required=False,
                   constraints=[
                       Custom(self.validate_medications, "Invalid medications")
                   ])
            .field("allergies", "LIST", required=False,
                   constraints=[
                       Custom(self.validate_allergies, "Invalid allergy information")
                   ])
            .field("lab_results", "LIST", required=False,
                   constraints=[
                       Custom(self.validate_lab_results, "Invalid lab results")
                   ])
        )
    
    def setup_medical_references(self):
        """Setup medical reference data"""
        
        # ICD-10 code patterns (simplified)
        self.icd10_pattern = r"^[A-Z]\d{2}\.?\d{0,3}$"
        
        # CPT code pattern
        self.cpt_pattern = r"^\d{5}$"
        
        # LOINC code pattern
        self.loinc_pattern = r"^\d{4,5}-\d$"
        
        # Valid medication routes
        self.valid_routes = {
            "PO", "IV", "IM", "SC", "SL", "PR", "TOP", "INH", "NASAL"
        }
        
        # Normal vital sign ranges
        self.vital_ranges = {
            "blood_pressure_systolic": (90, 180),
            "blood_pressure_diastolic": (60, 120),
            "heart_rate": (40, 180),
            "respiratory_rate": (8, 40),
            "temperature_f": (95.0, 105.0),
            "oxygen_saturation": (70, 100),
            "bmi": (10, 70)
        }
    
    def validate_not_ssn(self, patient_id: str) -> bool:
        """Ensure patient ID is not a SSN"""
        # Check if it looks like SSN (XXX-XX-XXXX or XXXXXXXXX)
        ssn_patterns = [
            r"^\d{3}-\d{2}-\d{4}$",
            r"^\d{9}$"
        ]
        
        for pattern in ssn_patterns:
            if re.match(pattern, patient_id):
                return False
        
        return True
    
    def validate_name_structure(self, name: Dict) -> bool:
        """Validate patient name structure"""
        required = ["first", "last"]
        
        if not all(field in name for field in required):
            return False
        
        # Check for PHI leakage in name fields
        for field, value in name.items():
            if isinstance(value, str):
                # Check name doesn't contain numbers (except suffixes)
                if field != "suffix" and re.search(r"\d", value):
                    return False
                
                # Check reasonable length
                if len(value) < 1 or len(value) > 50:
                    return False
        
        return True
    
    def validate_dob(self, dob: str) -> bool:
        """Validate date of birth"""
        try:
            birth_date = datetime.strptime(dob, "%Y-%m-%d").date()
            
            # Check not future date
            if birth_date > date.today():
                return False
            
            # Check reasonable age (0-150 years)
            age = (date.today() - birth_date).days / 365.25
            if age < 0 or age > 150:
                return False
            
            return True
        except:
            return False
    
    def validate_contact_info(self, contact: Dict) -> bool:
        """Validate contact information with HIPAA considerations"""
        
        # Phone validation
        if "phone" in contact:
            phone = re.sub(r'\D', '', contact["phone"])
            if len(phone) not in [10, 11]:
                return False
        
        # Email validation
        if "email" in contact:
            if not re.match(r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$", 
                           contact["email"]):
                return False
        
        # Address validation
        if "address" in contact:
            addr = contact["address"]
            required_addr = ["street", "city", "state", "zip"]
            if not all(field in addr for field in required_addr):
                return False
            
            # Validate ZIP code
            if not re.match(r"^\d{5}(-\d{4})?$", addr["zip"]):
                return False
        
        return True
    
    def validate_encounter_date(self, encounter_date: str) -> bool:
        """Validate encounter date"""
        try:
            enc_date = datetime.strptime(encounter_date, "%Y-%m-%d %H:%M:%S")
            
            # Check not future date
            if enc_date > datetime.now():
                return False
            
            # Check not too old (e.g., 100 years)
            if (datetime.now() - enc_date).days > 36500:
                return False
            
            return True
        except:
            return False
    
    def validate_vital_signs(self, vitals: Dict) -> bool:
        """Validate vital signs are within reasonable ranges"""
        
        for vital, value in vitals.items():
            if vital in self.vital_ranges:
                min_val, max_val = self.vital_ranges[vital]
                
                try:
                    numeric_value = float(value)
                    if numeric_value < min_val or numeric_value > max_val:
                        return False
                except:
                    return False
        
        # Check blood pressure format if present
        if "blood_pressure" in vitals:
            bp = vitals["blood_pressure"]
            if not re.match(r"^\d{2,3}/\d{2,3}$", bp):
                return False
            
            systolic, diastolic = map(int, bp.split("/"))
            if systolic <= diastolic:
                return False
        
        return True
    
    def validate_diagnoses(self, diagnoses: List) -> bool:
        """Validate diagnosis codes (ICD-10)"""
        
        if not diagnoses:
            return False
        
        for diagnosis in diagnoses:
            if isinstance(diagnosis, dict):
                code = diagnosis.get("code", "")
                
                # Validate ICD-10 format
                if not re.match(self.icd10_pattern, code):
                    return False
                
                # Check required fields
                if "description" not in diagnosis:
                    return False
            else:
                return False
        
        return True
    
    def validate_procedures(self, procedures: List) -> bool:
        """Validate procedure codes (CPT)"""
        
        for procedure in procedures:
            if isinstance(procedure, dict):
                code = procedure.get("code", "")
                
                # Validate CPT format
                if not re.match(self.cpt_pattern, code):
                    return False
                
                # Check date if present
                if "date" in procedure:
                    try:
                        proc_date = datetime.strptime(procedure["date"], "%Y-%m-%d")
                        if proc_date > datetime.now():
                            return False
                    except:
                        return False
            else:
                return False
        
        return True
    
    def validate_medications(self, medications: List) -> bool:
        """Validate medication information"""
        
        for med in medications:
            if not isinstance(med, dict):
                return False
            
            # Required fields
            if not all(field in med for field in ["name", "dosage", "route"]):
                return False
            
            # Validate route
            if med["route"] not in self.valid_routes:
                return False
            
            # Validate dosage format (simplified)
            dosage = med["dosage"]
            if not re.match(r"^\d+(\.\d+)?\s*(mg|mcg|g|mL|units?|IU)", dosage):
                return False
        
        return True
    
    def validate_allergies(self, allergies: List) -> bool:
        """Validate allergy information"""
        
        for allergy in allergies:
            if not isinstance(allergy, dict):
                return False
            
            # Required fields
            if not all(field in allergy for field in ["allergen", "reaction"]):
                return False
            
            # Validate severity if present
            if "severity" in allergy:
                if allergy["severity"] not in ["mild", "moderate", "severe", "life-threatening"]:
                    return False
        
        return True
    
    def validate_lab_results(self, lab_results: List) -> bool:
        """Validate laboratory results"""
        
        for result in lab_results:
            if not isinstance(result, dict):
                return False
            
            # Required fields
            if not all(field in result for field in ["test_name", "value", "unit", "reference_range"]):
                return False
            
            # Validate LOINC code if present
            if "loinc_code" in result:
                if not re.match(self.loinc_pattern, result["loinc_code"]):
                    return False
            
            # Check if value is numeric or a valid result
            value = result["value"]
            if not (isinstance(value, (int, float)) or 
                   value in ["positive", "negative", "pending", "abnormal"]):
                return False
        
        return True
    
    def perform_hipaa_compliance_check(self, record: Dict) -> Dict[str, Any]:
        """Check for HIPAA compliance issues"""
        
        issues = []
        compliant = True
        
        # Check for exposed PHI
        phi_fields = ["ssn", "social_security", "license_number", "full_account"]
        for field in phi_fields:
            if field in record:
                issues.append(f"Exposed PHI field: {field}")
                compliant = False
        
        # Check data minimization
        if "patient_id" in record and "mrn" in record:
            # Both identifiers present - check if necessary
            pass
        
        # Check encryption indicators (in real system)
        if "encryption_status" in record and not record["encryption_status"]:
            issues.append("Data not encrypted")
            compliant = False
        
        # Check audit trail
        if "audit_trail" not in record or not record["audit_trail"]:
            issues.append("Missing audit trail")
            compliant = False
        
        return {
            "compliant": compliant,
            "issues": issues,
            "risk_level": "low" if compliant else "high"
        }
    
    def validate_healthcare_record(self, record_data: Dict, 
                                  record_type: str = "patient") -> Dict[str, Any]:
        """Validate a complete healthcare record"""
        
        # Select schema
        if record_type == "clinical":
            schema = self.clinical_schema
        else:
            schema = self.patient_schema
        
        # Create record
        record = Record(data=record_data)
        
        # Schema validation
        result = schema.validate(record, coerce=True)
        
        if not result.valid:
            return {
                "valid": False,
                "errors": result.errors,
                "warnings": result.warnings
            }
        
        # HIPAA compliance check
        hipaa_check = self.perform_hipaa_compliance_check(result.value.data)
        
        # Data quality scoring
        quality_score = self.calculate_data_quality_score(result.value.data)
        
        return {
            "valid": True,
            "data": result.value.data,
            "hipaa_compliant": hipaa_check["compliant"],
            "hipaa_issues": hipaa_check["issues"],
            "quality_score": quality_score,
            "warnings": result.warnings
        }
    
    def calculate_data_quality_score(self, record: Dict) -> float:
        """Calculate data quality score (0-100)"""
        
        score = 100.0
        
        # Completeness check
        optional_fields = ["middle_name", "suffix", "secondary_phone", "emergency_contact"]
        for field in optional_fields:
            if field not in record or not record[field]:
                score -= 2  # Small penalty for missing optional fields
        
        # Consistency check
        if "dob" in record and "age" in record:
            calculated_age = (date.today() - datetime.strptime(record["dob"], "%Y-%m-%d").date()).days / 365.25
            if abs(calculated_age - record["age"]) > 1:
                score -= 10  # Inconsistent age
        
        # Timeliness check
        if "last_updated" in record:
            last_update = datetime.fromisoformat(record["last_updated"])
            days_old = (datetime.now() - last_update).days
            if days_old > 365:
                score -= min(20, days_old / 365 * 10)  # Stale data penalty
        
        return max(0, score)

# Usage example
def test_healthcare_validation():
    validator = HealthcareDataValidator()
    
    # Valid patient record
    valid_patient = {
        "patient_id": "PAT-1234567890",
        "mrn": "MRN-12345678",
        "name": {
            "first": "John",
            "last": "Doe",
            "middle": "A"
        },
        "dob": "1980-05-15",
        "gender": "M",
        "contact": {
            "phone": "555-123-4567",
            "email": "john.doe@example.com",
            "address": {
                "street": "123 Main St",
                "city": "Boston",
                "state": "MA",
                "zip": "02101"
            }
        },
        "audit_trail": True
    }
    
    result = validator.validate_healthcare_record(valid_patient, "patient")
    print(f"Patient validation: {result}")

if __name__ == "__main__":
    test_healthcare_validation()
```

## Best Practices Summary

1. **Layer validation** - Combine schema, business rules, and compliance checks
2. **Provide clear error messages** - Help users understand what needs correction
3. **Use appropriate constraints** - Match validation to actual requirements
4. **Consider performance** - Cache schemas and validation results when possible
5. **Maintain audit trails** - Track validation failures for analysis
6. **Implement graceful degradation** - Handle partial validation failures appropriately
7. **Version your schemas** - Track schema changes over time
8. **Test edge cases** - Validate with extreme and boundary values
9. **Monitor validation metrics** - Track success rates and common failures
10. **Document validation rules** - Maintain clear documentation of all validation requirements