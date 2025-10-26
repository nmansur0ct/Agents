# TPM Agentic Programming Cheat Sheet

## 🎯 **Quick Reference for 1-Hour Session**

---

## 📍 **File Locations**

| File | Purpose | Lines to Edit |
|------|---------|---------------|
| `config.py` | Business keywords & domain expertise | 42-85 |
| `nodes.py` | Agent prompts & collaboration logic | 98-125, 220-230 |
| `run.py` | CLI commands & testing | N/A (just run commands) |

---

## 🔧 **Exercise Quick Commands**

### **Exercise 1: Business Keywords**
```bash
# File: config.py, lines 42-85
# Add your keywords to keyword_mappings dictionary

# Test command:
python run.py --json '[{"name": "Test", "description": "your-keyword here"}]' --verbose
```

### **Exercise 2: Custom Agent**
```bash
# File: nodes.py, lines 98-125
# Modify the prompt starting with "You are an agent for..."

# Test command:
python run.py --json '[{"name": "Domain Feature", "description": "domain-specific feature"}]' --llm --verbose
```

### **Exercise 3: Agent Collaboration**
```bash
# File: nodes.py, lines 220-230
# Add collaboration notes in the llm_analysis section

# Test command:
python run.py --file samples/features.json --llm --verbose --auto-save
```

---

## 💼 **Business Domain Examples**

### **E-commerce Keywords**
```python
"cart-abandonment": {"revenue": 0.8, "reach": 0.7},
"personalization": {"revenue": 0.7, "complexity": 0.6},
"inventory": {"risk_reduction": 0.8, "engineering": 0.5},
"recommendations": {"revenue": 0.6, "complexity": 0.7},
"wishlist": {"reach": 0.6, "engineering": 0.3},
```

### **FinTech Keywords**
```python
"compliance": {"risk_reduction": 0.9, "engineering": 0.7},
"fraud-detection": {"risk_reduction": 0.8, "complexity": 0.7},
"kyc": {"risk_reduction": 0.8, "engineering": 0.6},
"plaid-integration": {"dependency": 0.7, "engineering": 0.5},
"pci-dss": {"risk_reduction": 0.9, "engineering": 0.8},
```

### **Healthcare Keywords**
```python
"hipaa": {"risk_reduction": 0.9, "engineering": 0.8},
"clinical-workflow": {"reach": 0.8, "complexity": 0.7},
"patient-portal": {"reach": 0.7, "engineering": 0.6},
"hl7-fhir": {"dependency": 0.8, "complexity": 0.8},
"telemedicine": {"reach": 0.8, "engineering": 0.7},
```

### **Manufacturing Keywords**
```python
"supply-chain": {"risk_reduction": 0.7, "complexity": 0.6},
"predictive-maintenance": {"revenue": 0.7, "complexity": 0.8},
"iot-sensors": {"dependency": 0.7, "engineering": 0.6},
"quality-control": {"risk_reduction": 0.8, "engineering": 0.5},
"lean-manufacturing": {"revenue": 0.6, "complexity": 0.5},
```

---

## 🤖 **Agent Personality Templates**

### **FinTech Agent**
```python
prompt = f"""You are an agent for financial technology feature analysis. Your expertise includes:
- Regulatory compliance (SOX, PCI-DSS, GDPR)
- Financial risk and fraud prevention
- Banking integration complexity
- Customer trust in financial services

Prioritize security, compliance, and customer trust.
Analyze: {feature_desc}"""
```

### **Healthcare Agent**
```python
prompt = f"""You are an agent for healthcare technology feature analysis. Your expertise includes:
- HIPAA compliance and patient data protection
- Clinical workflow integration
- Patient safety and care quality
- Healthcare system interoperability

Prioritize patient safety, compliance, and care quality.
Analyze: {feature_desc}"""
```

### **E-commerce Agent**
```python
prompt = f"""You are an agent for e-commerce feature analysis. Your expertise includes:
- Customer conversion and retention
- Shopping experience optimization
- Payment and checkout flows
- Inventory and supply chain

Prioritize customer experience, conversion rates, and operational efficiency.
Analyze: {feature_desc}"""
```

---

## ⚡ **Quick Test Commands**

### **Test Keywords Only**
```bash
python run.py --json '[{"name": "Keyword Test", "description": "test your-new-keyword functionality"}]' --verbose
```

### **Test LLM Agent**
```bash
python run.py --json '[{"name": "Agent Test", "description": "domain-specific feature description"}]' --llm --verbose
```

### **Test Full Pipeline**
```bash
python run.py --file samples/features.json --llm --verbose --auto-save
```

### **Quick Output Check**
```bash
# See just the prioritized results
python run.py --file samples/features.json --verbose | grep -A 5 "prioritized_features"

# See agent collaboration messages
python run.py --file samples/features.json --llm --verbose | grep -i "agent"
```

---

## 🔍 **What to Look For**

### **Exercise 1 Success Indicators**
- ✅ Your keywords appear in the "notes" section
- ✅ Factor scores change when your keywords are present
- ✅ Console shows "High [factor] inferred from 'your-keyword'"

### **Exercise 2 Success Indicators**
- ✅ LLM output shows domain-specific language
- ✅ Analysis reflects your agent's expertise area
- ✅ Factors align with your domain priorities

### **Exercise 3 Success Indicators**
- ✅ "Agent Override" messages appear in notes
- ✅ "Agent Collaboration" messages show transparency
- ✅ Decision changes are explained with rationale

---

## 🚨 **Common Mistakes**

### **Syntax Errors**
```python
# ❌ Wrong - Missing comma
"keyword1": {"revenue": 0.8}
"keyword2": {"revenue": 0.7}

# ✅ Correct - With comma
"keyword1": {"revenue": 0.8},
"keyword2": {"revenue": 0.7}
```

### **Factor Ranges**
```python
# ❌ Wrong - Values outside 0-1 range
"keyword": {"revenue": 1.5, "engineering": -0.2}

# ✅ Correct - Values in 0-1 range
"keyword": {"revenue": 0.9, "engineering": 0.2}
```

### **Case Sensitivity**
```python
# ❌ Wrong - Won't match "Machine Learning" in text
"Machine Learning": {"complexity": 0.8}

# ✅ Correct - Will match various cases
"machine-learning": {"complexity": 0.8}
```

---

## 📊 **Factor Meanings**

| Factor | Low (0.0-0.3) | Medium (0.4-0.6) | High (0.7-1.0) |
|--------|---------------|-------------------|-----------------|
| **reach** | Few users | Some users | Most/all users |
| **revenue** | No revenue impact | Moderate impact | Major revenue driver |
| **risk_reduction** | No risk benefit | Some risk mitigation | Critical risk reduction |
| **engineering** | Simple change | Moderate effort | Major development |
| **dependency** | No dependencies | Some integrations | Many complex deps |
| **complexity** | Straightforward | Some complexity | Very complex |

---

## 🎯 **Success Metrics**

After your 1-hour session, you should be able to:

- [ ] **Add 5 domain-specific keywords** that influence factor scoring
- [ ] **Create a specialized agent** with domain expertise
- [ ] **See agent collaboration** in the decision-making process  
- [ ] **Explain to your team** how agentic programming works
- [ ] **Identify opportunities** to apply agentic concepts in your organization

---

## 📞 **Help Commands**

### **Check Environment**
```bash
python --version  # Should be 3.8+
pip list | grep -E "(openai|click|pydantic)"  # Check dependencies
```

### **Reset to Clean State**
```bash
git checkout nodes.py config.py  # Reset your changes
git status  # Check what's modified
```

### **View Current Configuration**
```bash
python -c "from config import Config; print(Config.default().keyword_mappings)"
```

---

## 🚀 **Ready to Code?**

1. **Open your editor**: `code config.py` or `nano config.py`
2. **Start with Exercise 1**: Add your business keywords
3. **Test immediately**: Run the test command after each change
4. **Build confidence**: See your modifications work before moving on
5. **Document learnings**: Keep notes of what works for your domain

**Happy Agentic Programming!** 🤖✨