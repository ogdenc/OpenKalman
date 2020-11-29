var namespace_open_kalman_1_1internal =
[
    [ "detail", null, [
      [ "join_impl", "utils_8hpp.html#a0b33e31ff75398869a554f8eaca083d9", null ],
      [ "prepend_impl", "utils_8hpp.html#af80175047883842df251060a4fc80f1c", null ],
      [ "sqrt_impl", "utils_8hpp.html#a9f2a6f9071ac6fa7e918a4866dceb460", null ],
      [ "tuple_slice_impl", "utils_8hpp.html#a1a1e548edb04af8a677b3d64a442b8ae", null ]
    ] ],
    [ "CovarianceBase< Derived, ArgType, std::enable_if_t<((not square_root_covariance< Derived > and self_adjoint_matrix< ArgType >) or(square_root_covariance< Derived > and triangular_matrix< ArgType >)) and(not std::is_lvalue_reference_v< ArgType >) and(not internal::contains_nested_lvalue_reference< ArgType >)> >", "struct_open_kalman_1_1internal_1_1_covariance_base_3_01_derived_00_01_arg_type_00_01std_1_1enabl29df945888bdf150793bc6dc6b1445fb.html", "struct_open_kalman_1_1internal_1_1_covariance_base_3_01_derived_00_01_arg_type_00_01std_1_1enabl29df945888bdf150793bc6dc6b1445fb" ],
    [ "CovarianceBase< Derived, ArgType, std::enable_if_t<((not square_root_covariance< Derived > and self_adjoint_matrix< ArgType >) or(square_root_covariance< Derived > and triangular_matrix< ArgType >)) and(std::is_lvalue_reference_v< ArgType > or internal::contains_nested_lvalue_reference< ArgType >)> >", "struct_open_kalman_1_1internal_1_1_covariance_base_3_01_derived_00_01_arg_type_00_01std_1_1enabla13a6eb46bea1ac02dc83313a9b3010e.html", "struct_open_kalman_1_1internal_1_1_covariance_base_3_01_derived_00_01_arg_type_00_01std_1_1enabla13a6eb46bea1ac02dc83313a9b3010e" ],
    [ "CovarianceBase< Derived, ArgType, std::enable_if_t<(square_root_covariance< Derived > or not self_adjoint_matrix< ArgType >) and(not square_root_covariance< Derived > or not triangular_matrix< ArgType >) and(not std::is_lvalue_reference_v< ArgType >) and(not internal::contains_nested_lvalue_reference< ArgType >)> >", "struct_open_kalman_1_1internal_1_1_covariance_base_3_01_derived_00_01_arg_type_00_01std_1_1enabl33dea91a16b54869feea364852fa5cb9.html", "struct_open_kalman_1_1internal_1_1_covariance_base_3_01_derived_00_01_arg_type_00_01std_1_1enabl33dea91a16b54869feea364852fa5cb9" ],
    [ "CovarianceBase< Derived, ArgType, std::enable_if_t<(square_root_covariance< Derived > or not self_adjoint_matrix< ArgType >) and(not square_root_covariance< Derived > or not triangular_matrix< ArgType >) and(std::is_lvalue_reference_v< ArgType > or internal::contains_nested_lvalue_reference< ArgType >)> >", "struct_open_kalman_1_1internal_1_1_covariance_base_3_01_derived_00_01_arg_type_00_01std_1_1enablf7ccd8f72f32390098ce960e2476ff72.html", "struct_open_kalman_1_1internal_1_1_covariance_base_3_01_derived_00_01_arg_type_00_01std_1_1enablf7ccd8f72f32390098ce960e2476ff72" ],
    [ "CovarianceBaseBase", "struct_open_kalman_1_1internal_1_1_covariance_base_base.html", null ],
    [ "default_split_function", "struct_open_kalman_1_1internal_1_1default__split__function.html", null ],
    [ "is_atomic_coefficient_group", "struct_open_kalman_1_1internal_1_1is__atomic__coefficient__group.html", null ],
    [ "is_cholesky_form", "struct_open_kalman_1_1internal_1_1is__cholesky__form.html", null ],
    [ "is_composite_coefficients", "struct_open_kalman_1_1internal_1_1is__composite__coefficients.html", null ],
    [ "is_covariance_nestable", "struct_open_kalman_1_1internal_1_1is__covariance__nestable.html", null ],
    [ "is_diagonal_matrix", "struct_open_kalman_1_1internal_1_1is__diagonal__matrix.html", null ],
    [ "is_element_gettable", "struct_open_kalman_1_1internal_1_1is__element__gettable.html", null ],
    [ "is_element_settable", "struct_open_kalman_1_1internal_1_1is__element__settable.html", null ],
    [ "is_equivalent_to", "struct_open_kalman_1_1internal_1_1is__equivalent__to.html", null ],
    [ "is_identity_matrix", "struct_open_kalman_1_1internal_1_1is__identity__matrix.html", null ],
    [ "is_lower_triangular_matrix", "struct_open_kalman_1_1internal_1_1is__lower__triangular__matrix.html", null ],
    [ "is_prefix_of", "struct_open_kalman_1_1internal_1_1is__prefix__of.html", null ],
    [ "is_self_adjoint_matrix", "struct_open_kalman_1_1internal_1_1is__self__adjoint__matrix.html", null ],
    [ "is_self_contained", "struct_open_kalman_1_1internal_1_1is__self__contained.html", null ],
    [ "is_typed_matrix_nestable", "struct_open_kalman_1_1internal_1_1is__typed__matrix__nestable.html", null ],
    [ "is_upper_triangular_matrix", "struct_open_kalman_1_1internal_1_1is__upper__triangular__matrix.html", null ],
    [ "is_zero_matrix", "struct_open_kalman_1_1internal_1_1is__zero__matrix.html", null ],
    [ "LinearTransformBase", "struct_open_kalman_1_1internal_1_1_linear_transform_base.html", "struct_open_kalman_1_1internal_1_1_linear_transform_base" ],
    [ "MatrixBase", "struct_open_kalman_1_1internal_1_1_matrix_base.html", "struct_open_kalman_1_1internal_1_1_matrix_base" ],
    [ "TypedMatrixBase", "struct_open_kalman_1_1internal_1_1_typed_matrix_base.html", "struct_open_kalman_1_1internal_1_1_typed_matrix_base" ],
    [ "constexpr_sqrt", "namespace_open_kalman_1_1internal.html#a0de6ea03b16d31991db9843df173d674", null ],
    [ "convert_nested_matrix", "namespace_open_kalman_1_1internal.html#aa3bb16dc004f421a9f872f5683ca985d", null ],
    [ "ElementSetter", "namespace_open_kalman_1_1internal.html#a115fec3bd8019b3095d35e2d4a992099", null ],
    [ "ElementSetter", "namespace_open_kalman_1_1internal.html#a690fd8a900f4f0f55e0a6c8db13115d8", null ],
    [ "ElementSetter", "namespace_open_kalman_1_1internal.html#a0813ee5d8fad4463af82eddab22aad4f", null ],
    [ "ElementSetter", "namespace_open_kalman_1_1internal.html#a5bd347f057423b3bbcc33b294abf1c49", null ],
    [ "ElementSetter", "namespace_open_kalman_1_1internal.html#a5f27a50978bc678b048c9b1548a1ef34", null ],
    [ "ElementSetter", "namespace_open_kalman_1_1internal.html#acd53f3e5c19bd2ce35a29eda03f067e6", null ],
    [ "get_perturbation", "namespace_open_kalman_1_1internal.html#af91d60d85b47d8da8e456e80b4e1e1e1", null ],
    [ "join", "namespace_open_kalman_1_1internal.html#ab4c55d1d16722622d8e01a474f69faea", null ],
    [ "make_ElementSetter", "namespace_open_kalman_1_1internal.html#a4f1054af4c6652fcf660dcc3cfc502db", null ],
    [ "make_ElementSetter", "namespace_open_kalman_1_1internal.html#a9620519424597b39a847bce7e570d2b3", null ],
    [ "prepend", "namespace_open_kalman_1_1internal.html#a8acbe50bdd93f93904cb245c9f981db1", null ],
    [ "tuple_replicate", "namespace_open_kalman_1_1internal.html#a1eb076c1ccda659ec9f6b9e955bf2dae", null ],
    [ "tuple_slice", "namespace_open_kalman_1_1internal.html#a0fbebd42d3419522aec582110074b5dc", null ],
    [ "contains_nested_lvalue_reference", "namespace_open_kalman_1_1internal.html#a0f2240261b5fa68286e374b29a8a6283", null ],
    [ "same_triangle_type_as", "namespace_open_kalman_1_1internal.html#a35106ced3df898a6d4917e86b57df608", null ],
    [ "transformation_args", "namespace_open_kalman_1_1internal.html#ab8a56a0c90a852f2323f7b4cb015b190", null ]
];