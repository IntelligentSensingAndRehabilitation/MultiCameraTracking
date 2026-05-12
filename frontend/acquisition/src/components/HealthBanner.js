import React from 'react';
import { useContext } from 'react';
import { Link } from 'react-router-dom';
import { AcquisitionState } from '../AcquisitionApi';

const HealthBanner = () => {
    const { healthReport } = useContext(AcquisitionState);

    if (!healthReport) return null;
    if (healthReport.overall !== 'error') return null;

    const errorFindings = (healthReport.findings || []).filter(f => f.level === 'error');
    if (errorFindings.length === 0) return null;

    const top = errorFindings[0];
    const extra = errorFindings.length > 1 ? ` (+${errorFindings.length - 1} more)` : '';

    return (
        <div style={{
            background: '#c0392b',
            color: 'white',
            padding: '8px 16px',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'center',
            fontWeight: '500',
        }}>
            <span>{top.message}{extra}</span>
            <Link
                to="/diagnostics"
                style={{ color: 'white', textDecoration: 'underline', fontWeight: 'bold' }}
            >
                Open Diagnostics →
            </Link>
        </div>
    );
};

export default HealthBanner;
