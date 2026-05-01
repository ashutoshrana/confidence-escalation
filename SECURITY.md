# Security Policy

## Supported Versions

| Version | Supported |
|---------|-----------|
| 0.1.x   | Yes       |

## Reporting a Vulnerability

**Do not report security vulnerabilities through public GitHub issues.**

Use the [GitHub Security Advisory](../../security/advisories/new) feature,
or email the maintainer directly.

You should receive a response within 72 hours.

Please include:
- Description of the vulnerability
- Steps to reproduce
- Potential impact
- Suggested fix (if any)

## Scope

This library implements confidence-gated escalation middleware for LLM agents.
The security surface includes:

- **Confidence score manipulation**: Adversarial inputs that artificially inflate
  or deflate confidence scores to bypass escalation thresholds
- **Handler injection**: Ensuring escalation handlers cannot be hijacked to
  suppress human oversight (relevant to EU AI Act Article 14)
- **Audit log integrity**: Compliance logging handlers must not expose PII
  in structured log output
- **Threshold policy bypass**: Inputs engineered to avoid triggering escalation
  when they should (OWASP Agentic AI ASI-09: Human-Agent Trust Exploitation)

This library does **not** manage authentication, network access, or
cryptography directly. Integrating applications are responsible for securing
LLM API keys and human review queue access.

## Disclosure Policy

- Confirmation within 72 hours
- Initial assessment within 7 days
- Patch target within 30 days of confirmed vulnerability
- Credit given in release notes unless anonymity preferred
