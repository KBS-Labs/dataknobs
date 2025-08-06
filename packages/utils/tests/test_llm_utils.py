import datetime

from dataknobs_utils import llm_utils


def test_get_value_by_key():
    assert llm_utils.get_value_by_key({"foo": {"bar": "baz"}}, "foo.bar", "Undefined") == "baz"
    assert (
        llm_utils.get_value_by_key({"foo": {"bar": "baz"}}, "foo.baz", "Undefined") == "Undefined"
    )
    assert llm_utils.get_value_by_key({"foo": {"bar": "baz"}}, "bar", None) is None


def _do_prompt_message_test(pm):
    assert str(pm) == '{"role": "system", "content": "You are a helpful assistant."}'
    assert (
        pm.to_json()
        == '{"role": "system", "content": "You are a helpful assistant.", "metadata": {"metadata": "test"}}'
    )
    assert pm.get_message() == {"role": "system", "content": "You are a helpful assistant."}


def test_prompt_message():
    pm = llm_utils.PromptMessage(
        "system",
        "You are a helpful assistant.",
        {"metadata": "test"},
    )
    _do_prompt_message_test(pm)


def test_prompt_message_build_instance():
    pm = llm_utils.PromptMessage.build_instance(
        {
            "role": "system",
            "content": "You are a helpful assistant.",
            "metadata": {"metadata": "test"},
        }
    )
    _do_prompt_message_test(pm)


def test_prompt_tree_construction():
    pt = llm_utils.PromptTree(role="1", content="1", metadata={"id": 1})
    assert (
        pt.message.to_json(with_metadata=True)
        == '{"role": "1", "content": "1", "metadata": {"id": 1}}'
    )

    pt2 = llm_utils.PromptTree(message=pt.message)
    assert (
        pt2.message.to_json(with_metadata=True)
        == '{"role": "1", "content": "1", "metadata": {"id": 1}}'
    )

    pt_with_role_override = llm_utils.PromptTree(message=pt.message, role="2")
    assert (
        pt_with_role_override.message.to_json(with_metadata=True)
        == '{"role": "2", "content": "1", "metadata": {"id": 1}}'
    )

    pt_with_content_override = llm_utils.PromptTree(message=pt.message, content="2")
    assert (
        pt_with_content_override.message.to_json(with_metadata=True)
        == '{"role": "1", "content": "2", "metadata": {"id": 1}}'
    )

    pt_with_metadata_override = llm_utils.PromptTree(message=pt.message, metadata={"id": 2})
    assert (
        pt_with_metadata_override.message.to_json(with_metadata=True)
        == '{"role": "1", "content": "1", "metadata": {"id": 2}}'
    )


def test_prompt_tree_serialization_roundtrip():
    #
    # Build a tree like this:
    #
    # (1 (2 (4 5)) (3 (6 (7))))
    #
    # aka:
    #
    # 1 -- 2 -- 4
    #  \     \_ 5
    #   \_ 3 -- 6 -- 7
    #
    root_pt = llm_utils.PromptTree(role="1", content="1", metadata={"id": 1})
    pt1 = root_pt.add_message(role="2", content="2", metadata={"id": 2})
    pt2 = root_pt.add_message(role="3", content="3", metadata={"id": 3})
    pt1_1 = pt1.add_message(role="4", content="4", metadata={"id": 4})
    pt1_2 = pt1.add_message(role="5", content="5", metadata={"id": 5})
    pt2_1 = pt2.add_message(role="6", content="6", metadata={"id": 6})
    pt2_1_1 = pt2_1.add_message(role="7", content="7", metadata=None)
    data = pt2_1_1.serialize_tree(full=True)  # serialize

    re_pt = llm_utils.PromptTree.build_instance(data)  # deserialize/reconstruct
    data2 = re_pt.serialize_tree(full=True)  # reserialize

    assert data == data2
    assert re_pt.message.to_json(with_metadata=True) == pt2_1_1.message.to_json(with_metadata=True)
    assert pt2_1_1.node_id == re_pt.node_id
    assert pt2_1_1.node_count == re_pt.node_count
    assert pt2_1_1.depth == 3
    assert re_pt.depth == 3
    assert pt2_1_1.metadata == {"id": 6}  # inherited
    assert re_pt.metadata == {"id": 6}  # inherited
    assert pt2_1_1.get_metadata_value("id") == 6  # inherited
    assert re_pt.get_metadata_value("id") == 6  # inherited
    assert re_pt.get_messages(level_offset=-1, with_metadata=False) == [
        {"role": "1", "content": "1"},
        {"role": "3", "content": "3"},
        {"role": "6", "content": "6"},
        {"role": "7", "content": "7"},
    ]
    assert pt2_1_1.get_level_node(0) == pt2_1_1  # self
    assert pt2_1_1.get_level_node(1) == pt2_1  # parent
    assert pt2_1_1.get_level_node(2) == pt2  # grandparent
    assert pt2_1_1.get_level_node(3) == root_pt  # great grandparent
    assert pt2_1_1.get_level_node(10) == root_pt  # capped
    assert pt2_1_1.get_level_node(-1) == root_pt  # root
    assert pt2_1_1.get_level_node(-2) == pt2  # root+1
    assert pt2_1_1.get_level_node(-3) == pt2_1  # root+2
    assert pt2_1_1.get_level_node(-4) == pt2_1_1  # root+3
    assert pt2_1_1.get_level_node(-10) == pt2_1_1  # capped
    assert pt2_1_1.get_messages(0, False) == [{"role": "7", "content": "7"}]
    assert pt2_1_1.get_messages(1, False) == [
        {"role": "6", "content": "6"},
        {"role": "7", "content": "7"},
    ]
    assert pt2_1_1.get_duration() == 0  # no recorded times
    assert pt2_1_1.find_node_by_id(3) == pt1_1  # "4" has id 3
    n67 = pt2_1_1.find_nodes(lambda node: node.data.get_metadata_value("id") >= 6)
    assert n67 == [pt2_1, pt2_1_1]

    # serialize just pt2_1_1
    data7 = pt2_1_1.serialize_tree(full=False)
    re_pt2 = llm_utils.PromptTree.build_instance(data7)
    assert re_pt2.message.to_json(with_metadata=True) == re_pt.message.to_json(with_metadata=True)
    assert re_pt2.node_id != re_pt.node_id
    assert re_pt2.depth == 0


def test_get_duration_and_apply():
    #
    # Build a tree like this:
    #
    # (1 (2 (4 5)) (3 (6 (7))))
    #
    # aka:
    #
    # 1 -- 2 -- 4
    #  \     \_ 5
    #   \_ 3 -- 6 -- 7
    #
    # Where time duration at 2, 3, and 6 is 10s each
    #
    reftime = datetime.datetime.now()
    tm0 = reftime.isoformat()
    tm10 = (reftime - datetime.timedelta(seconds=10)).isoformat()
    tm20 = (reftime - datetime.timedelta(seconds=20)).isoformat()
    tm30 = (reftime - datetime.timedelta(seconds=30)).isoformat()

    root_pt = llm_utils.PromptTree(role="1", content="1", metadata={"id": 1})
    pt1 = root_pt.add_message(
        role="2",
        content="2",
        metadata={
            "id": 3,
            "execution_data": {
                "starttime": tm30,
                "endtime": tm20,
            },
        },
    )
    pt2 = root_pt.add_message(
        role="3",
        content="3",
        metadata={
            "id": 3,
            "execution_data": {
                "starttime": tm20,
                "endtime": tm10,
            },
        },
    )
    pt1_1 = pt1.add_message(role="4", content="4", metadata={"id": 4})
    pt1_2 = pt1.add_message(role="5", content="5", metadata={"id": 5})
    pt2_1 = pt2.add_message(
        role="6",
        content="6",
        metadata={
            "id": 3,
            "execution_data": {
                "starttime": tm10,
                "endtime": tm0,
            },
        },
    )
    pt2_1_1 = pt2_1.add_message(role="7", content="7", metadata=None)

    assert pt2_1_1.get_duration(0) == 0
    # import pdb; pdb.set_trace()
    # stophere = True
    assert pt2_1_1.get_duration(1) == 10
    assert pt2_1_1.get_duration(2) == 20
    assert pt2_1_1.get_duration(-1) == 20
    assert pt1_2.get_duration(-1) == 10

    # Using the example MessageCollector to test PromptTree.apply
    mc = llm_utils.MessageCollector(with_metadata=False)
    pt2_1_1.apply(mc, level_offset=-1)
    messages_from_apply = mc.messages
    messages_from_pt = pt2_1_1.get_messages(-1, False)
    assert messages_from_apply == messages_from_pt
