#!/usr/bin/env python3
"""
Data validation pipeline example using FSM with custom functions.

This example demonstrates a validation pipeline that:
1. Validates multiple fields in sequence
2. Collects all validation results
3. Routes to different end states based on validation outcome
4. Preserves detailed error information for debugging
"""

import re
from typing import Dict, Any
from dataknobs_fsm.api.simple import SimpleFSM


def normalize_input(state) -> Dict[str, Any]:
    """Normalize input data and initialize validation tracking."""
    data = state.data.copy()
    
    # Initialize validation results tracking
    data['validations'] = {
        'email': {'checked': False, 'valid': False, 'errors': []},
        'phone': {'checked': False, 'valid': False, 'errors': []},
        'age': {'checked': False, 'valid': False, 'errors': []}
    }
    
    # Normalize email
    if 'email' in data:
        data['email'] = data['email'].strip().lower()
    
    # Normalize phone (remove non-digits)
    if 'phone' in data:
        data['phone'] = re.sub(r'\D', '', data['phone'])
    
    return data


def validate_email(state) -> Dict[str, Any]:
    """Validate email and update validation results."""
    data = state.data.copy()
    email = data.get('email', '')
    
    validation = data['validations']['email']
    validation['checked'] = True
    
    if not email:
        validation['errors'].append("Email is required")
    else:
        # Check format
        pattern = r'^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$'
        if not re.match(pattern, email):
            validation['errors'].append(f"Invalid email format: {email}")
        
        # Check domain
        allowed_domains = ['gmail.com', 'yahoo.com', 'outlook.com', 'company.com']
        if '@' in email:
            domain = email.split('@')[1]
            if domain not in allowed_domains:
                validation['errors'].append(f"Domain '{domain}' is not in allowed list: {allowed_domains}")
    
    validation['valid'] = len(validation['errors']) == 0
    
    return data


def validate_phone(state) -> Dict[str, Any]:
    """Validate phone number and update validation results."""
    data = state.data.copy()
    phone = data.get('phone', '')
    
    validation = data['validations']['phone']
    validation['checked'] = True
    
    if not phone:
        validation['errors'].append("Phone number is required")
    elif len(phone) != 10:
        validation['errors'].append(f"Phone must be 10 digits, got {len(phone)}")
    elif not phone.isdigit():
        validation['errors'].append("Phone must contain only digits")
    
    validation['valid'] = len(validation['errors']) == 0
    
    # Format phone if valid
    if validation['valid']:
        data['phone_formatted'] = f"({phone[:3]}) {phone[3:6]}-{phone[6:]}"
    
    return data


def validate_age(state) -> Dict[str, Any]:
    """Validate age and update validation results."""
    data = state.data.copy()
    age = data.get('age')
    
    validation = data['validations']['age']
    validation['checked'] = True
    
    if age is None:
        validation['errors'].append("Age is required")
    elif not isinstance(age, int):
        validation['errors'].append(f"Age must be an integer, got {type(age).__name__}")
    elif age < 0:
        validation['errors'].append(f"Age must be non-negative, got {age}")
    elif age > 150:
        validation['errors'].append(f"Age must be realistic (<=150), got {age}")
    elif age < 18:
        validation['errors'].append(f"Must be 18 or older, got {age}")
    
    validation['valid'] = len(validation['errors']) == 0
    
    return data


def calculate_validation_summary(state) -> Dict[str, Any]:
    """Calculate overall validation status and prepare summary."""
    data = state.data.copy()
    
    validations = data['validations']
    
    # Count valid fields
    valid_count = sum(1 for v in validations.values() if v['valid'])
    total_count = len(validations)
    
    # Collect all errors
    all_errors = []
    for field, validation in validations.items():
        for error in validation['errors']:
            all_errors.append(f"{field}: {error}")
    
    # Determine overall status
    if valid_count == total_count:
        data['validation_status'] = 'fully_valid'
        data['validation_message'] = "All validations passed"
    elif valid_count > 0:
        data['validation_status'] = 'partially_valid'
        data['validation_message'] = f"{valid_count}/{total_count} validations passed"
    else:
        data['validation_status'] = 'invalid'
        data['validation_message'] = "All validations failed"
    
    data['validation_errors'] = all_errors
    data['validation_score'] = valid_count / total_count if total_count > 0 else 0
    
    return data


# FSM configuration for validation pipeline
config = {
    "name": "DataValidationPipeline",
    "main_network": "main",
    "data_mode": {
        "default": "direct"
    },
    "networks": [{
        "name": "main",
        "states": [
            {
                "name": "input",
                "is_start": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "email": {"type": "string"},
                        "phone": {"type": "string"},
                        "age": {"type": "integer"}
                    }
                }
            },
            {
                "name": "normalize",
                "functions": {
                    "transform": {
                        "type": "registered",
                        "name": "normalize_input"
                    }
                }
            },
            {
                "name": "validate_email",
                "functions": {
                    "transform": {
                        "type": "registered",
                        "name": "validate_email"
                    }
                }
            },
            {
                "name": "validate_phone", 
                "functions": {
                    "transform": {
                        "type": "registered",
                        "name": "validate_phone"
                    }
                }
            },
            {
                "name": "validate_age",
                "functions": {
                    "transform": {
                        "type": "registered",
                        "name": "validate_age"
                    }
                }
            },
            {
                "name": "summarize",
                "functions": {
                    "transform": {
                        "type": "registered",
                        "name": "calculate_validation_summary"
                    }
                }
            },
            {
                "name": "fully_valid",
                "is_end": True
            },
            {
                "name": "partially_valid",
                "is_end": True
            },
            {
                "name": "invalid",
                "is_end": True
            }
        ],
        "arcs": [
            {"from": "input", "to": "normalize"},
            {"from": "normalize", "to": "validate_email"},
            {"from": "validate_email", "to": "validate_phone"},
            {"from": "validate_phone", "to": "validate_age"},
            {"from": "validate_age", "to": "summarize"},
            {
                "from": "summarize",
                "to": "fully_valid",
                "condition": {
                    "type": "inline",
                    "code": "data.get('validation_status') == 'fully_valid'"
                }
            },
            {
                "from": "summarize",
                "to": "partially_valid",
                "condition": {
                    "type": "inline",
                    "code": "data.get('validation_status') == 'partially_valid'"
                }
            },
            {
                "from": "summarize",
                "to": "invalid",
                "condition": {
                    "type": "inline",
                    "code": "data.get('validation_status') == 'invalid'"
                }
            }
        ]
    }]
}


def main():
    """Run the validation pipeline example."""
    # Create FSM with custom functions
    fsm = SimpleFSM(
        config,
        custom_functions={
            'normalize_input': normalize_input,
            'validate_email': validate_email,
            'validate_phone': validate_phone,
            'validate_age': validate_age,
            'calculate_validation_summary': calculate_validation_summary
        }
    )
    
    # Test cases covering different validation scenarios
    test_cases = [
        {
            "name": "All Valid",
            "data": {
                "email": "john.doe@gmail.com",
                "phone": "(555) 123-4567",
                "age": 25
            }
        },
        {
            "name": "Invalid Email Domain",
            "data": {
                "email": "user@invalid.com",
                "phone": "5551234567",
                "age": 30
            }
        },
        {
            "name": "Invalid Phone",
            "data": {
                "email": "user@yahoo.com",
                "phone": "123",
                "age": 22
            }
        },
        {
            "name": "Underage",
            "data": {
                "email": "teen@gmail.com",
                "phone": "5551234567",
                "age": 16
            }
        },
        {
            "name": "Multiple Issues",
            "data": {
                "email": "not-an-email",
                "phone": "abc",
                "age": -5
            }
        },
        {
            "name": "Missing Fields",
            "data": {}
        }
    ]
    
    print("Data Validation Pipeline Example")
    print("=" * 70)
    
    for test_case in test_cases:
        print(f"\nTest Case: {test_case['name']}")
        print(f"Input: {test_case['data']}")
        print("-" * 50)
        
        result = fsm.process(test_case['data'])
        
        if result['success']:
            print(f"✓ Processing succeeded")
            print(f"Final State: {result['final_state']}")
            print(f"Path: {' -> '.join(result['path'])}")
            
            final_data = result['data']
            if 'validation_status' in final_data:
                print(f"\nValidation Status: {final_data['validation_status']}")
                print(f"Message: {final_data['validation_message']}")
                print(f"Score: {final_data['validation_score']:.1%}")
                
                if final_data['validation_errors']:
                    print("\nErrors Found:")
                    for error in final_data['validation_errors']:
                        print(f"  • {error}")
                
                print("\nField Results:")
                for field, validation in final_data['validations'].items():
                    status = "✓" if validation['valid'] else "✗"
                    print(f"  {status} {field}: {'Valid' if validation['valid'] else 'Invalid'}")
        else:
            print(f"✗ Processing failed")
            print(f"Error: {result.get('error', 'Unknown error')}")
    
    print("\n" + "=" * 70)
    print("Example complete!")


if __name__ == "__main__":
    main()