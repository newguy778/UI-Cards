// RefreshVA.test.js

import { RefreshVA } from './RefreshVA';
import RasaApi from '../../services/api/RasaServices';
import { ToggleRefreshVA, IncreaseMessageCounter, AddMsg } from './index';
import ResponseStatus from '../../Util/ResponseStatus';
import { ActivityState } from './ActivityState';
import { Unavailable } from '../../const/activity';
import MessageObjGen from '../../Util/ObjGen';
import { ErrorMessage401 } from '../../const';
import RasaQuery from '../../Util/RasaQuery';

// Mock dependencies
jest.mock('../../services/api/RasaServices');
jest.mock('./index', () => ({
  ToggleRefreshVA: jest.fn(),
  IncreaseMessageCounter: jest.fn(),
  AddMsg: jest.fn(),
}));
jest.mock('../../Util/ResponseStatus');
jest.mock('./ActivityState');
jest.mock('../../Util/ObjGen');
jest.mock('../../Util/RasaQuery');

describe('RefreshVA', () => {
  let dispatch;

  beforeEach(() => {
    dispatch = jest.fn();
    jest.clearAllMocks();
  });

  it('should handle empty message_contents by dispatching ErrorMessage401', async () => {
    const payload = {};
    const response = {
      data: {
        message_contents: [],
      },
    };

    RasaApi.getResponseMessage.mockResolvedValue(response);
    MessageObjGen.messageBubbleGenerator.mockReturnValue('messageBubble');
    ResponseStatus.responseStatusCheck.mockImplementation((res, callback) => {
      callback();
      return null;
    });

    await RefreshVA(payload)(dispatch);

    expect(dispatch).toHaveBeenCalledWith(ToggleRefreshVA(true));
    expect(ResponseStatus.responseStatusCheck).toHaveBeenCalledWith(response, expect.any(Function));
    expect(dispatch).toHaveBeenCalledWith(IncreaseMessageCounter());
    expect(MessageObjGen.messageBubbleGenerator).toHaveBeenCalledWith('VA', 'TEXT', ErrorMessage401);
    expect(dispatch).toHaveBeenCalledWith(AddMsg('messageBubble'));
  });

  it('should handle non-empty message_contents by dispatching explored response', async () => {
    const payload = {};
    const messageContents = ['message1', 'message2'];
    const response = {
      data: {
        message_contents: messageContents,
      },
    };

    RasaApi.getResponseMessage.mockResolvedValue(response);
    RasaQuery.exploreVAResponse.mockReturnValue('exploredResponse');
    ResponseStatus.responseStatusCheck.mockImplementation((res, callback) => {
      callback();
      return null;
    });

    await RefreshVA(payload)(dispatch);

    expect(dispatch).toHaveBeenCalledWith(ToggleRefreshVA(true));
    expect(ResponseStatus.responseStatusCheck).toHaveBeenCalledWith(response, expect.any(Function));
    expect(dispatch).toHaveBeenCalledWith(IncreaseMessageCounter());
    expect(RasaQuery.exploreVAResponse).toHaveBeenCalledWith(messageContents);
    expect(dispatch).toHaveBeenCalledWith(AddMsg('exploredResponse'));
  });

  it('should handle truthy status by dispatching status and ActivityState', async () => {
    const payload = {};
    const response = {};
    const status = 'someStatus';

    RasaApi.getResponseMessage.mockResolvedValue(response);
    ResponseStatus.responseStatusCheck.mockReturnValue(status);

    await RefreshVA(payload)(dispatch);

    expect(dispatch).toHaveBeenCalledWith(ToggleRefreshVA(true));
    expect(ResponseStatus.responseStatusCheck).toHaveBeenCalledWith(response, expect.any(Function));
    expect(dispatch).toHaveBeenCalledWith(IncreaseMessageCounter());
    expect(dispatch).toHaveBeenCalledWith(AddMsg(status));
    expect(dispatch).toHaveBeenCalledWith(ActivityState(Unavailable));
  });

  it('should handle errors from RasaApi.getResponseMessage', async () => {
    const payload = {};
    const error = new Error('Network Error');

    RasaApi.getResponseMessage.mockRejectedValue(error);

    await expect(RefreshVA(payload)(dispatch)).rejects.toThrow('Network Error');
  });
});
