import React from 'react';
import { useContext, useState } from 'react';
import { Link } from 'react-router-dom';
import { AcquisitionState } from '../AcquisitionApi';

// Red banner for error-level signals from the health report (persistent
// state, self-clearing) and live diagnostic envelopes (events, dismissable).
const HealthBanner = () => {
    const { healthReport, sessionInsights } = useContext(AcquisitionState);
    const [dismissedTs, setDismissedTs] = useState(null);

    const errorInsights = (sessionInsights || []).filter(
        (e) => e.level === 'error'
    );
    const freshInsights = errorInsights.filter(
        (e) => !dismissedTs || !e.ts || e.ts > dismissedTs
    );
    const latestInsight = freshInsights[freshInsights.length - 1];

    const errorFindings = (healthReport && healthReport.findings)
        ? healthReport.findings.filter((f) => f.level === 'error')
        : [];
    const persistentError = errorFindings[0];

    if (!latestInsight && !persistentError) return null;

    // Live envelopes (recent events) take precedence over persistent findings.
    const showingInsight = !!latestInsight;
    const source = showingInsight ? latestInsight : persistentError;
    const message = source.message;
    const remediation = showingInsight
        ? (source.details && source.details.remediation) || []
        : source.remediation || [];
    const extra =
        !showingInsight && errorFindings.length > 1
            ? ` (+${errorFindings.length - 1} more)`
            : '';

    return (
        <div style={{
            background: '#c0392b',
            color: 'white',
            padding: '8px 16px',
            display: 'flex',
            justifyContent: 'space-between',
            alignItems: 'flex-start',
            gap: '16px',
            fontWeight: '500',
        }}>
            <div style={{ flexGrow: 1 }}>
                <div>{message}{extra}</div>
                {remediation.length > 0 && (
                    <ol style={{
                        margin: '4px 0 0 20px',
                        padding: 0,
                        fontSize: '0.9em',
                        fontWeight: 'normal',
                    }}>
                        {remediation.map((step, i) => (
                            <li key={i}>{step}</li>
                        ))}
                    </ol>
                )}
            </div>
            <div style={{ display: 'flex', gap: '12px', flexShrink: 0 }}>
                <Link
                    to="/diagnostics"
                    style={{ color: 'white', textDecoration: 'underline', fontWeight: 'bold' }}
                >
                    Open Diagnostics →
                </Link>
                {showingInsight && (
                    <button
                        onClick={() => setDismissedTs(latestInsight.ts || new Date().toISOString())}
                        style={{
                            background: 'transparent',
                            color: 'white',
                            border: '1px solid white',
                            borderRadius: '3px',
                            padding: '2px 8px',
                            cursor: 'pointer',
                            fontSize: '0.85em',
                        }}
                    >
                        Dismiss
                    </button>
                )}
            </div>
        </div>
    );
};

export default HealthBanner;
