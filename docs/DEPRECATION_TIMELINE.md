# Dataknobs Deprecation Timeline

## Overview

This document outlines the deprecation timeline for the legacy `dataknobs` package as we transition to the modular architecture.

## Timeline

### Phase 1: Soft Launch (Current - Month 3)
**Status: Active**

- ✅ New modular packages available (`dataknobs-structures`, `dataknobs-utils`, `dataknobs-xization`)
- ✅ Legacy package updated with compatibility layer (v0.0.15)
- ✅ Documentation and migration guides published
- ⏳ Gather user feedback and address issues

### Phase 2: Active Migration (Months 3-6)
**Target: Q2 2025**

- Legacy package shows deprecation warnings on import
- Active communication to users about migration
- Support both old and new import patterns
- Regular updates to modular packages based on feedback
- Migration assistance for major users

### Phase 3: Deprecation Warning (Months 6-9)
**Target: Q3 2025**

- Legacy package (v0.0.16) shows prominent deprecation warnings
- No new features added to legacy package
- Security and critical bug fixes only
- Final push for user migration
- Documentation prominently features modular packages

### Phase 4: End of Support (Month 12)
**Target: Q4 2025**

- Legacy package marked as deprecated on PyPI
- Final version (v0.0.17) with end-of-life notice
- No further updates to legacy package
- Redirect all users to modular packages
- Archive legacy package code

## Version Roadmap

| Version | Release Date | Status | Notes |
|---------|-------------|--------|-------|
| 0.0.15 | Current | Released | Compatibility layer with new packages |
| 0.0.16 | Month 6 | Planned | Add deprecation warnings |
| 0.0.17 | Month 12 | Planned | Final release with EOL notice |

## Communication Plan

### Immediate Actions
1. Update PyPI project description
2. Add banner to legacy package README
3. Create GitHub issue for migration tracking

### Month 3
- Blog post about modular architecture benefits
- Email to known major users
- Social media announcements

### Month 6
- Deprecation warning in package
- Update all documentation
- Final migration guide updates

### Month 9
- Last call announcements
- Direct outreach to remaining users
- Prepare sunset documentation

### Month 12
- End of life announcement
- Archive repository
- Final documentation update

## Support Commitments

### During Migration Period (Months 1-12)
- **Bug Fixes**: Critical and security fixes only
- **Features**: No new features in legacy package
- **Documentation**: Maintained but not expanded
- **Support**: Active migration assistance

### Post-Deprecation (Month 12+)
- **Bug Fixes**: None
- **Features**: None
- **Documentation**: Archived, read-only
- **Support**: Redirect to modular packages

## Success Metrics

- **Month 3**: 25% of users migrated
- **Month 6**: 60% of users migrated
- **Month 9**: 90% of users migrated
- **Month 12**: 100% migration complete

## Risk Mitigation

### Risk: Slow adoption of new packages
**Mitigation**: 
- Provide automated migration tools
- Offer migration support
- Maintain compatibility layer longer if needed

### Risk: Breaking changes for users
**Mitigation**:
- Extensive testing of compatibility layer
- Gradual deprecation with warnings
- Clear migration documentation

### Risk: Unknown dependencies
**Mitigation**:
- Monitor PyPI download stats
- GitHub dependency graph analysis
- Direct user surveys

## Contact

For migration assistance or questions:
- GitHub Issues: https://github.com/kbs-labs/dataknobs/issues
- Migration Guide: [docs/migration-guide.md](migration-guide.md)
