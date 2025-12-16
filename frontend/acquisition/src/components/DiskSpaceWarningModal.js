import React, { useContext } from 'react';
import { Modal, Button, Alert } from 'react-bootstrap';
import { AcquisitionState } from '../AcquisitionApi';

const DiskSpaceWarningModal = () => {
    const {
        diskSpaceInfo,
        showDiskWarningModal,
        setShowDiskWarningModal,
        diskWarningOnStartup
    } = useContext(AcquisitionState);

    const handleClose = () => setShowDiskWarningModal(false);

    if (!diskSpaceInfo) {
        return null;
    }

    return (
        <Modal
            show={showDiskWarningModal}
            onHide={diskWarningOnStartup ? undefined : handleClose}
            backdrop={diskWarningOnStartup ? "static" : true}
            keyboard={!diskWarningOnStartup}
        >
            <Modal.Header closeButton={!diskWarningOnStartup}>
                <Modal.Title>
                    {diskWarningOnStartup ? "Warning: Low Disk Space" : "Low Disk Space"}
                </Modal.Title>
            </Modal.Header>
            <Modal.Body>
                <Alert variant="warning">
                    <Alert.Heading>Disk Space Running Low</Alert.Heading>
                    <p>
                        The data directory has limited storage space remaining:
                    </p>
                    <ul>
                        <li><strong>Available:</strong> {diskSpaceInfo.disk_space_gb_remaining} GB ({diskSpaceInfo.disk_space_percent_remaining}%)</li>
                        <li><strong>Total:</strong> {diskSpaceInfo.disk_space_total_gb} GB</li>
                    </ul>
                    <hr />
                    {diskWarningOnStartup ? (
                        <p className="mb-0">
                            <strong>Please free up disk space before starting a recording session.</strong>
                            Recording with insufficient space may result in incomplete or corrupted data.
                        </p>
                    ) : (
                        <p className="mb-0">
                            The recording will proceed, but you may want to free up space soon to avoid
                            filling the disk during this session.
                        </p>
                    )}
                </Alert>
            </Modal.Body>
            <Modal.Footer>
                {diskWarningOnStartup ? (
                    <Button variant="primary" onClick={handleClose}>
                        I Understand
                    </Button>
                ) : (
                    <>
                        <Button variant="secondary" onClick={handleClose}>
                            Continue Recording
                        </Button>
                    </>
                )}
            </Modal.Footer>
        </Modal>
    );
};

export default DiskSpaceWarningModal;
