import React, { useContext, useEffect, useState } from 'react';
import { Container, Card, Badge, Button, ListGroup, Accordion, Table } from 'react-bootstrap';
import { AcquisitionState } from '../AcquisitionApi';

const severityVariant = {
    ok: 'success',
    warn: 'warning',
    error: 'danger',
    unknown: 'secondary',
};

const SeverityBadge = ({ level }) => (
    <Badge bg={severityVariant[level] || 'secondary'} style={{ textTransform: 'uppercase' }}>
        {level || 'unknown'}
    </Badge>
);

const FindingList = ({ findings }) => {
    if (!findings || findings.length === 0) {
        return <p className="text-muted mb-0">No findings.</p>;
    }
    return (
        <ListGroup variant="flush">
            {findings.map((f, i) => (
                <ListGroup.Item key={`${f.code}-${i}`} className="d-flex justify-content-between align-items-start">
                    <div className="ms-2 me-auto">
                        <div className="fw-bold">{f.message}</div>
                        <small className="text-muted">{f.code}</small>
                    </div>
                    <SeverityBadge level={f.level} />
                </ListGroup.Item>
            ))}
        </ListGroup>
    );
};

const SubsystemCard = ({ title, severity, findings, extra }) => (
    <Card className="mb-3">
        <Card.Header className="d-flex justify-content-between align-items-center">
            <strong>{title}</strong>
            <SeverityBadge level={severity} />
        </Card.Header>
        <Card.Body>
            {extra}
            <FindingList findings={findings} />
        </Card.Body>
    </Card>
);

const HostHealthPanel = () => {
    const { healthReport, fetchHealth } = useContext(AcquisitionState);
    const [refreshing, setRefreshing] = useState(false);

    const handleRefresh = async () => {
        setRefreshing(true);
        try {
            await fetchHealth(true);
        } finally {
            setRefreshing(false);
        }
    };

    if (!healthReport) {
        return (
            <Card className="mb-4">
                <Card.Header><strong>Host Health</strong></Card.Header>
                <Card.Body>
                    <p className="text-muted">Loading health report…</p>
                    <Button onClick={handleRefresh} disabled={refreshing} size="sm">
                        {refreshing ? 'Checking…' : 'Re-check now'}
                    </Button>
                </Card.Body>
            </Card>
        );
    }

    const { overall, dhcp, cameras, host_network, recording_state, deployment_mode, generated_at } = healthReport;

    return (
        <Card className="mb-4">
            <Card.Header className="d-flex justify-content-between align-items-center">
                <div>
                    <strong>Host Health</strong>{' '}
                    <SeverityBadge level={overall} />
                </div>
                <Button onClick={handleRefresh} disabled={refreshing} size="sm" variant="outline-primary">
                    {refreshing ? 'Checking…' : 'Re-check now'}
                </Button>
            </Card.Header>
            <Card.Body>
                <div className="mb-3 text-muted small">
                    Mode: <code>{deployment_mode}</code> &nbsp;|&nbsp;
                    Recording state: <code>{recording_state}</code> &nbsp;|&nbsp;
                    Last checked: {generated_at ? new Date(generated_at).toLocaleString() : '—'}
                </div>

                <SubsystemCard
                    title="DHCP server"
                    severity={dhcp?.findings?.length ? maxLevel(dhcp.findings) : 'ok'}
                    findings={dhcp?.findings}
                    extra={
                        dhcp?.applicable ? (
                            <Table size="sm" borderless className="mb-2">
                                <tbody>
                                    <tr><td>Service</td><td>{describeBool(dhcp.service_active)}</td></tr>
                                    <tr><td>Interface IP</td><td><code>{dhcp.interface_ip || '—'}</code></td></tr>
                                    <tr><td>Active leases</td><td>{dhcp.lease_count}</td></tr>
                                </tbody>
                            </Table>
                        ) : (
                            <p className="text-muted small mb-2">DHCP checks skipped (laptop mode).</p>
                        )
                    }
                />

                <SubsystemCard
                    title="Cameras"
                    severity={cameras?.findings?.length ? maxLevel(cameras.findings) : 'ok'}
                    findings={cameras?.findings}
                    extra={
                        <Table size="sm" borderless className="mb-2">
                            <tbody>
                                <tr><td>Expected</td><td>{cameras?.expected?.length ?? 0}</td></tr>
                                <tr><td>Detected</td><td>{cameras?.detected?.length ?? 0}</td></tr>
                                <tr><td>Missing</td><td>{cameras?.missing?.join(', ') || '—'}</td></tr>
                                <tr><td>Extra</td><td>{cameras?.extra?.join(', ') || '—'}</td></tr>
                            </tbody>
                        </Table>
                    }
                />

                <SubsystemCard
                    title="Host network"
                    severity={host_network?.findings?.length ? maxLevel(host_network.findings) : 'ok'}
                    findings={host_network?.findings}
                    extra={
                        <Table size="sm" borderless className="mb-2">
                            <tbody>
                                <tr><td>Interface</td><td><code>{host_network?.interface}</code></td></tr>
                                <tr><td>Carrier</td><td>{describeBool(host_network?.carrier_up)}</td></tr>
                                <tr><td>MTU</td><td>{host_network?.mtu ?? '—'} (expected {host_network?.expected_mtu})</td></tr>
                                <tr><td>rmem_max</td><td>{host_network?.rmem_max ?? '—'} (expected {host_network?.expected_rmem_max})</td></tr>
                            </tbody>
                        </Table>
                    }
                />
            </Card.Body>
        </Card>
    );
};

const CurrentSessionPanel = () => {
    const { sessionSummary, sessionInsights, fetchSessionSummary } = useContext(AcquisitionState);
    const [refreshing, setRefreshing] = useState(false);

    useEffect(() => {
        fetchSessionSummary();
        // eslint-disable-next-line react-hooks/exhaustive-deps
    }, []);

    const handleRefresh = async () => {
        setRefreshing(true);
        try {
            await fetchSessionSummary();
        } finally {
            setRefreshing(false);
        }
    };

    return (
        <Card className="mb-4">
            <Card.Header className="d-flex justify-content-between align-items-center">
                <strong>Current Session</strong>
                <Button onClick={handleRefresh} disabled={refreshing} size="sm" variant="outline-primary">
                    {refreshing ? 'Refreshing…' : 'Refresh summary'}
                </Button>
            </Card.Header>
            <Card.Body>
                {!sessionSummary && (
                    <p className="text-muted">No active session, or no trials yet.</p>
                )}

                {sessionSummary && (
                    <>
                        <div className="mb-3 small text-muted">
                            {sessionSummary.n_trials} trials analyzed.{' '}
                            Last refreshed: {new Date(sessionSummary.generated_at).toLocaleTimeString()}
                        </div>

                        <Accordion defaultActiveKey={["0", "1", "2"]} alwaysOpen>
                            <Accordion.Item eventKey="0">
                                <Accordion.Header>
                                    Insights ({sessionSummary.insights?.length || 0})
                                </Accordion.Header>
                                <Accordion.Body>
                                    {renderStringList(sessionSummary.insights, 'No session-level patterns detected.')}
                                </Accordion.Body>
                            </Accordion.Item>

                            <Accordion.Item eventKey="1">
                                <Accordion.Header>
                                    Recommendations ({sessionSummary.recommendations?.length || 0})
                                </Accordion.Header>
                                <Accordion.Body>
                                    {renderStringList(sessionSummary.recommendations, 'No recommendations.')}
                                </Accordion.Body>
                            </Accordion.Item>

                            <Accordion.Item eventKey="2">
                                <Accordion.Header>
                                    Per-trial findings ({sessionSummary.trial_findings?.length || 0})
                                </Accordion.Header>
                                <Accordion.Body>
                                    {renderStringList(sessionSummary.trial_findings, 'No per-trial issues detected.')}
                                </Accordion.Body>
                            </Accordion.Item>
                        </Accordion>
                    </>
                )}

                {sessionInsights && sessionInsights.length > 0 && (
                    <Card className="mt-3">
                        <Card.Header><strong>Live trial events</strong></Card.Header>
                        <Card.Body>
                            <ListGroup variant="flush">
                                {sessionInsights.map((ev, i) => (
                                    <ListGroup.Item key={i} className="d-flex justify-content-between align-items-start">
                                        <div className="ms-2 me-auto">
                                            <div className="fw-bold">{ev.message}</div>
                                            <small className="text-muted">
                                                {ev.code}{ev.ts ? ` · ${new Date(ev.ts).toLocaleTimeString()}` : ''}
                                            </small>
                                        </div>
                                        <SeverityBadge level={ev.level} />
                                    </ListGroup.Item>
                                ))}
                            </ListGroup>
                        </Card.Body>
                    </Card>
                )}
            </Card.Body>
        </Card>
    );
};

const DiagnosticsPage = () => (
    <Container className="mt-3">
        <h3 className="mb-3">Diagnostics</h3>
        <HostHealthPanel />
        <CurrentSessionPanel />
    </Container>
);

const describeBool = (b) => (b === true ? 'OK' : b === false ? 'down' : '—');

const SEVERITY_RANK = { unknown: 0, ok: 1, warn: 2, error: 3 };
const maxLevel = (findings) => {
    if (!findings || findings.length === 0) return 'ok';
    return findings.reduce(
        (acc, f) => (SEVERITY_RANK[f.level] > SEVERITY_RANK[acc] ? f.level : acc),
        'ok'
    );
};

const renderStringList = (items, emptyMessage) => {
    if (!items || items.length === 0) {
        return <p className="text-muted mb-0">{emptyMessage}</p>;
    }
    return (
        <ListGroup variant="flush">
            {items.map((s, i) => (
                <ListGroup.Item key={i}>{s}</ListGroup.Item>
            ))}
        </ListGroup>
    );
};

export default DiagnosticsPage;
