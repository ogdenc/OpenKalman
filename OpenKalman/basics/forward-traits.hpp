/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2019-2021 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Forward declarations for traits relating to OpenKalman or native matrix types.
 */

#ifndef OPENKALMAN_FORWARD_TRAITS_HPP
#define OPENKALMAN_FORWARD_TRAITS_HPP

#include <type_traits>


/**
 * \namespace OpenKalman
 * \brief The root namespace for OpenKalman.
 *
 * \internal
 * \namespace OpenKalman::internal
 * \brief Namespace for internal definitions, not intended for use outside of OpenKalman development.
 */


namespace OpenKalman
{

  // ---------------- //
  //  max_indices_of  //
  // ---------------- //

  /**
   * \brief The maximum number of indices of structure T.
   * \tparam T A tensor (vector, matrix, etc.)
   */
  template<typename T>
  struct max_indices_of
    : std::integral_constant<std::size_t, interface::IndexibleObjectTraits<std::decay_t<T>>::max_indices> {};


  /**
   * \brief helper template for \ref max_indices_of.
   */
  template<typename T>
  static constexpr std::size_t max_indices_of_v = max_indices_of<T>::value;


  // ---------------- //
  //  scalar_type_of  //
  // ---------------- //

  /**
   * \brief Type scalar type (e.g., std::float, std::double, std::complex<double>) of a tensor or index descriptor.
   * \tparam T A matrix, expression, array, or \ref index_descriptor.
   * \internal \sa interface::IndexibleObjectTraits
   */
#ifdef __cpp_concepts
  template<typename T>
#else
  template<typename T, typename = void>
#endif
  struct scalar_type_of {};


#ifdef __cpp_concepts
  template<typename T> requires requires {typename interface::IndexibleObjectTraits<std::decay_t<T>>::scalar_type; }
  struct scalar_type_of<T>
#else
  template<typename T>
  struct scalar_type_of<T, std::void_t<typename interface::IndexibleObjectTraits<std::decay_t<T>>::scalar_type>>
#endif
  {
    using type = typename interface::IndexibleObjectTraits<std::decay_t<T>>::scalar_type;
  };


  /**
   * \brief helper template for \ref scalar_type_of.
   */
  template<typename T>
  using scalar_type_of_t = typename scalar_type_of<T>::type;


  // ----------- //
  //  indexible  //
  // ----------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_indexible : std::false_type {};

    template<typename T>
    struct is_indexible<T, std::enable_if_t<(max_indices_of<T>::value >= 0) and scalar_type<typename scalar_type_of<T>::type>>>
      : std::true_type {};
  }
#endif


  /**
   * \brief T is a generalized tensor type.
   * \details T can be a tensor over a vector space, but can also be an analogous algebraic structure over a
   * tensor product of modules over division rings (e.g., an vector-like structure that contains angles).
   */
  template<typename T>
#ifdef __cpp_concepts
  concept indexible = (max_indices_of_v<T> >= 0) and scalar_type<scalar_type_of_t<T>>;
#else
  constexpr bool indexible = detail::is_indexible<T>::value;
#endif


  // -------------------- //
  //  index_dimension_of  //
  // -------------------- //

  /**
   * \brief The dimension of an index for a matrix, expression, or array.
   * \details The static constexpr <code>value</code> member indicates the size of the object associated with a
   * particular index. If the dimension is dynamic, <code>value</code> will be \ref dynamic_size.
   * \tparam N The index
   * \tparam T The matrix, expression, or array
   * \internal \sa interface::IndexTraits
   */
#ifdef __cpp_concepts
  template<indexible T, std::size_t N> requires (N < max_indices_of_v<T>)
  struct index_dimension_of
    : std::integral_constant<std::size_t, interface::IndexTraits<std::decay_t<T>>::template dimension<N>> {};
#else
  template<typename T, std::size_t N = 0, typename = void>
  struct index_dimension_of {};

  template<typename T, std::size_t N>
  struct index_dimension_of<T, N, std::enable_if_t<indexible<T> and N < max_indices_of<T>::value>>
    : std::integral_constant<std::size_t, interface::IndexTraits<std::decay_t<T>>::template dimension<N>> {};
#endif


  /**
   * \brief helper template for \ref index_dimension_of.
   */
  template<typename T, std::size_t N = 0>
  static constexpr auto index_dimension_of_v = index_dimension_of<T, N>::value;


  // ------------------ //
  //  row_dimension_of  //
  // ------------------ //

  /**
   * \brief The row dimension of a matrix, expression, or array.
   * \note If the row dimension is dynamic, then <code>value</code> is \ref dynamic_size.
   * \tparam T The matrix, expression, or array.
   * \internal \sa interface::IndexTraits
   */
  template<typename T>
  using row_dimension_of = index_dimension_of<T, 0>;


  /**
   * \brief helper template for \ref row_dimension_of.
   */
  template<typename T>
  static constexpr auto row_dimension_of_v = row_dimension_of<T>::value;


  // --------------------- //
  //  column_dimension_of  //
  // --------------------- //

  /**
   * \brief The column dimension of a matrix, expression, or array.
   * \note If the column dimension is dynamic, then <code>value</code> is \ref dynamic_size.
   * \tparam T The matrix, expression, or array.
   * \internal \sa interface::IndexTraits
   */
  template<typename T>
  using column_dimension_of = index_dimension_of<T, 1>;


  /**
   * \brief helper template for \ref column_dimension_of.
   */
  template<typename T>
  static constexpr auto column_dimension_of_v = column_dimension_of<T>::value;


  // ------------------- //
  //  dynamic_dimension  //
  // ------------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, std::size_t N, typename = void>
    struct is_dynamic_dimension : std::false_type {};

    template<typename T, std::size_t N>
    struct is_dynamic_dimension<T, N, std::enable_if_t<indexible<T> and N < max_indices_of<T>::value and
      index_dimension_of<T, N>::value == dynamic_size>>
      : std::true_type {};
  }
#endif


  /**
   * \brief Specifies that T's index N has a dimension defined at run time.
   * \details The matrix library interface will specify this for native matrices and expressions.
   */
  template<typename T, std::size_t N>
#ifdef __cpp_concepts
  concept dynamic_dimension = (N < max_indices_of_v<T>) and (index_dimension_of_v<T, N> == dynamic_size);
#else
  constexpr bool dynamic_dimension = detail::is_dynamic_dimension<T, N>::value;
#endif


  // -------------- //
  //  dynamic_rows  //
  // -------------- //

  /**
   * \brief Specifies that T has a row dimension that is defined at run time.
   * \details The matrix library interface will specify this for native matrices and expressions.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept dynamic_rows =
#else
  constexpr bool dynamic_rows =
#endif
    dynamic_dimension<T, 0>;


  // ----------------- //
  //  dynamic_columns  //
  // ----------------- //

  /**
   * \brief Specifies that T has a column dimension that is defined at run time.
   * \details The matrix library interface will specify this for native matrices and expressions.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept dynamic_columns =
#else
  constexpr bool dynamic_columns =
#endif
    dynamic_dimension<T, 1>;


  // --------------------------- //
  //  number_of_dynamic_indices  //
  // --------------------------- //

  namespace detail
  {
    template<typename T, std::size_t...I>
    constexpr std::size_t number_of_dynamic_indices_impl(std::index_sequence<I...>)
    {
      return ((dynamic_dimension<T, I> ? 1 : 0) + ... + 0);
    }
  }


  /**
   * \brief Counts the number of indices of T in which the dimensions are dynamic.
   */
#ifdef __cpp_concepts
  template<indexible T>
#else
  template<typename T>
#endif
  struct number_of_dynamic_indices : std::integral_constant<std::size_t,
    detail::number_of_dynamic_indices_impl<T>(std::make_index_sequence<max_indices_of_v<T>> {})> {};


  /**
   * \brief Helper template for \ref number_of_dynamic_indices
   */
#ifdef __cpp_concepts
  template<indexible T>
#else
  template<typename T>
#endif
  static constexpr std::size_t number_of_dynamic_indices_v = number_of_dynamic_indices<T>::value;


  // ------------------------ //
  //  has_dynamic_dimensions  //
  // ------------------------ //

  /**
   * \brief Specifies that T has at least one index with dynamic dimensions.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept has_dynamic_dimensions =
#else
  constexpr bool has_dynamic_dimensions =
#endif
    (number_of_dynamic_indices_v<T> > 0);


  // --------------------- //
  //  max_tensor_order_of  //
  // --------------------- //

  namespace detail
  {
    template<std::size_t i, typename T>
    constexpr std::size_t max_tensor_order_of_impl()
    {
      if constexpr (i == 0)
        return 0;
      else if constexpr (interface::IndexTraits<std::decay_t<T>>::template dimension<i - 1> == 0)
        return 0;
      else if constexpr (interface::IndexTraits<std::decay_t<T>>::template dimension<i - 1> == 1)
        return max_tensor_order_of_impl<i - 1, T>();
      else
        return 1 + max_tensor_order_of_impl<i - 1, T>();
    }
  }

  /**
   * \brief The maximum number of indices of structure T of size greater than 1 (including any dynamic indices).
   * \details If there are any indices of dimension 0, the result will be 0.
   * \tparam T A tensor (vector, matrix, etc.)
   */
  template<typename T>
  struct max_tensor_order_of
    : std::integral_constant<std::size_t, detail::max_tensor_order_of_impl<max_indices_of_v<T>, T>()> {};


  /**
   * \brief helper template for \ref max_indices_of.
   */
  template<typename T>
  static constexpr std::size_t max_tensor_order_of_v = max_tensor_order_of<T>::value;


  // ----------------------------------------------------------------------------- //
  //  coefficient_types_of, row_coefficient_types_of, column_coefficient_types_of  //
  // ----------------------------------------------------------------------------- //

  /**
   * \brief The coefficient types of T for index N.
   * \tparam T A matrix, expression, or array
   * \tparam N The index number (0 = rows, 1 = columns, etc.)
   */
#ifdef __cpp_concepts
  template<typename T, std::size_t N>
#else
  template<typename T, std::size_t N, typename = void>
#endif
  struct coefficient_types_of
  {
    using type = Dimensions<index_dimension_of_v<T, N>>;
  };


#ifdef __cpp_concepts
  template<indexible T, std::size_t N> requires (N < max_indices_of_v<T>) and
    requires { typename interface::CoordinateSystemTraits<std::decay_t<T>>::template coordinate_system_types<N>; }
  struct coefficient_types_of<T, N>
#else
  template<typename T, std::size_t N>
  struct coefficient_types_of<T, N, std::enable_if_t<indexible<T> and N < max_indices_of_v<T> and
    std::is_void<std::void_t<typename interface::CoordinateSystemTraits<std::decay_t<T>>::template coordinate_system_types<N>>>::value>>
#endif
  {
    using type = typename interface::CoordinateSystemTraits<std::decay_t<T>>::template coordinate_system_types<N>;
  };


  /**
   * \brief helper template for \ref coefficient_types_of.
   */
  template<typename T, std::size_t N>
  using coefficient_types_of_t = typename coefficient_types_of<T, N>::type;



  template<typename T>
  using row_coefficient_types_of = coefficient_types_of<T, 0>;


  /**
   * \brief helper template for \ref row_coefficient_types_of.
   */
  template<typename T>
  using row_coefficient_types_of_t = typename row_coefficient_types_of<T>::type;


  template<typename T>
  using column_coefficient_types_of = coefficient_types_of<T, 1>;


  /**
   * \brief helper template for \ref column_coefficient_types_of.
   */
  template<typename T>
  using column_coefficient_types_of_t = typename column_coefficient_types_of<T>::type;


  // --------------------- //
  //   has_untyped_index   //
  // --------------------- //

  /**
   * \brief Specifies that T has an untyped index N.
   * \details Index N of T is Euclidean and non-modular (e.g., Axis, TypedIndex<Axis, Axis>, etc.).
   */
#ifdef __cpp_concepts
  template<typename T, std::size_t N>
  concept has_untyped_index =
#else
  template<typename T, std::size_t N>
  constexpr bool has_untyped_index =
#endif
    euclidean_index_descriptor<coefficient_types_of_t<T, N>>;


  // ----------------------------------- //
  //   all_fixed_indices_are_euclidean   //
  // ----------------------------------- //

  namespace detail
  {
    template<typename T, std::size_t...I>
    constexpr bool all_fixed_indices_are_euclidean_impl(std::index_sequence<I...>) {
      return ((dynamic_dimension<T, I> or has_untyped_index<T, I>) and ...); }
  }


  /**
   * \brief Specifies that every fixed-size index of T is euclidean.
   * \details No fixed_size index of T is modular (e.g., Angle, Polar, Spherical, etc.).
   */
#ifdef __cpp_concepts
  template<typename T>
  concept all_fixed_indices_are_euclidean =
#else
  template<typename T>
  constexpr bool all_fixed_indices_are_euclidean =
#endif
    indexible<T> and (detail::all_fixed_indices_are_euclidean_impl<T>(std::make_index_sequence<max_indices_of_v<T>> {}));


  // ------------------------------- //
  //  maybe_index_descriptors_match  //
  // ------------------------------- //

  namespace detail
  {
    template<typename...>
    constexpr bool maybe_index_descriptors_match_impl(std::index_sequence<>) { return true; }

    template<std::size_t I, std::size_t...Is>
    constexpr bool maybe_index_descriptors_match_impl(std::index_sequence<I, Is...>) { return true; }

    template<typename T, typename...Ts, std::size_t I, std::size_t...Is>
    constexpr bool maybe_index_descriptors_match_impl(std::index_sequence<I, Is...>)
    {
      if constexpr (dynamic_dimension<T, I>)
        return maybe_index_descriptors_match_impl<Ts...>(std::index_sequence<I, Is...>{}) and
          maybe_index_descriptors_match_impl<T, Ts...>(std::index_sequence<Is...>{});
      else
        return ((dynamic_dimension<Ts, I> or equivalent_to<coefficient_types_of_t<T, I>, coefficient_types_of_t<Ts, I>>)
          and ... and maybe_index_descriptors_match_impl<T, Ts...>(std::index_sequence<Is...>{}));
    }
  }


  /**
   * \brief Specifies that all index descriptors of zero or more objects might be equivalent.
   */
#ifdef __cpp_concepts
  template<typename...Ts>
  concept maybe_index_descriptors_match =
#else
  template<typename...Ts>
  constexpr bool maybe_index_descriptors_match =
#endif
    (indexible<Ts> and ...) and
    (detail::maybe_index_descriptors_match_impl<Ts...>(std::make_index_sequence<std::max({max_indices_of_v<Ts>...})> {}));


  // ------------------------- //
  //  index_descriptors_match  //
  // ------------------------- //

  namespace detail
  {
    template<typename...Ts, std::size_t...Is>
    constexpr bool index_descriptors_match_impl(std::index_sequence<Is...>)
    {
      return ([](auto I){
        return equivalent_to<coefficient_types_of_t<Ts, decltype(I)::value>...>;
      }(std::integral_constant<std::size_t, Is>{}) and ...);
    }
  }


  /**
   * \brief Specifies that all index descriptors of zero or more objects are known at compile time to be equivalent.
   */
#ifdef __cpp_concepts
  template<typename...Ts>
  concept index_descriptors_match =
#else
  template<typename...Ts>
  constexpr bool index_descriptors_match =
#endif
    (indexible<Ts> and ...) and
    (detail::index_descriptors_match_impl<Ts...>(std::make_index_sequence<std::max({max_indices_of_v<Ts>...})> {}));


  // ------------------------------------- //
  //   compatible_with_index_descriptors   //
  // ------------------------------------- //

  namespace detail
  {
    template<typename T, std::size_t N, typename...Ds>
    struct compatible_with_index_descriptors_impl : std::false_type {};

    template<typename T, std::size_t N>
    struct compatible_with_index_descriptors_impl<T, N> : std::bool_constant<N == max_indices_of_v<T>> {};

    template<typename T, std::size_t N, typename D, typename...Ds>
    struct compatible_with_index_descriptors_impl<T, N, D, Ds...> : std::bool_constant<
#ifdef __cpp_concepts
      (dynamic_dimension<T, N> or equivalent_to<D, coefficient_types_of_t<T, N>>) and
#else
      (dynamic_dimension<T, std::min(N, max_indices_of_v<T> - 1)> or equivalent_to<D, coefficient_types_of_t<T, std::min(N, max_indices_of_v<T> - 1)>>) and
#endif
        compatible_with_index_descriptors_impl<T, N + 1, Ds...>::value> {};
  }


  /**
   * \brief \ref indexible T is compatible with \ref index_descriptor set Ds.
   */
  template<typename T, typename...Ds>
#ifdef __cpp_concepts
  concept compatible_with_index_descriptors =
#else
  constexpr bool compatible_with_index_descriptors =
#endif
    (max_indices_of_v<T> == sizeof...(Ds)) and (index_descriptor<Ds> and ...) and
    detail::compatible_with_index_descriptors_impl<T, 0, Ds...>::value;


  // ------------- //
  //   wrappable   //
  // ------------- //

  namespace detail
  {
    template<typename T, std::size_t...I>
    constexpr bool wrappable_impl(std::index_sequence<I...>) {
      return ((dynamic_dimension<T, I> or has_untyped_index<T, I + 1>) and ...); }
#ifndef __cpp_concepts

    template<typename T, typename = void>
    struct is_wrappable : std::false_type {};

    template<typename T>
    struct is_wrappable<T, std::enable_if_t<indexible<T>>>
      : std::bool_constant<(max_indices_of_v<T> >= 1) and
      (detail::wrappable_impl<T>(std::make_index_sequence<max_indices_of_v<T> - 1> {}))> {};
#endif
  }


  /**
   * \brief Specifies that every fixed-size index of T (other than potentially index 0) is euclidean.
   * \details This indicates that T is suitable for wrapping along index 0.
   * \sa get_wrappable
   */
  template<typename T>
#ifdef __cpp_concepts
  concept wrappable =
    (max_indices_of_v<T> >= 1) and (detail::wrappable_impl<T>(std::make_index_sequence<max_indices_of_v<T> - 1> {}));
#else
  constexpr bool wrappable = detail::is_wrappable<T>::value;
#endif


  // ------------------ //
  //  element_gettable  //
  // ------------------ //

#ifndef __cpp_lib_concepts
  namespace detail
  {
    template<typename T, typename = void, typename...I>
    struct element_gettable_impl : std::false_type {};

    template<typename T, typename...I>
    struct element_gettable_impl<T, std::void_t<
        decltype(interface::GetElement<std::decay_t<T>>::get(std::declval<T>(), static_cast<const std::size_t>(std::declval<I>())...))>, I...>
      : std::true_type {};
  }
#endif


  /**
   * \brief Specifies that a type has elements that can be retrieved with indices I... (of type std::size_t).
   * \details This concept should include anything for which get_element(...) is properly defined
   * \sa get_element
   */
  template<typename T, typename...I>
#ifdef __cpp_lib_concepts
  concept element_gettable = (sizeof...(I) > 0) and (sizeof...(I) <= max_indices_of_v<T>) and
    (std::convertible_to<I, const std::size_t> and ...) and
    requires(T t, I...i) { interface::GetElement<std::decay_t<T>>::get(t, static_cast<const std::size_t>(i)...); };
#else
  constexpr bool element_gettable = (sizeof...(I) > 0) and (std::is_convertible<I, const std::size_t>::value and ...) and
    detail::element_gettable_impl<T, void, I...>::value;
#endif


  // ------------------ //
  //  element_settable  //
  // ------------------ //

#ifndef __cpp_lib_concepts
  namespace detail
  {
    template<typename T, typename = void, typename...I>
    struct element_settable_impl : std::false_type {};

    template<typename T, typename...I>
    struct element_settable_impl<T, std::enable_if_t<(sizeof...(I) <= max_indices_of_v<T>) and
        std::is_same<decltype(interface::SetElement<std::decay_t<T>>::set(std::declval<T&&>(),
          std::declval<const typename scalar_type_of<T>::type&>(), static_cast<const std::size_t>(std::declval<I>())...)), T&&>::value>, I...>
      : std::true_type {};
  }
#endif


  /**
   * \brief Specifies that a type has elements that can be set with indices I... (of type std::size_t).
   * \details This concept should include anything for which set_element(...) is properly defined
   * \sa set_element
   */
  template<typename T, typename...I>
#ifdef __cpp_lib_concepts
  concept element_settable = (sizeof...(I) > 0) and (sizeof...(I) <= max_indices_of_v<T>) and
    (std::convertible_to<I, const std::size_t> and ...) and (not std::is_const_v<std::remove_reference_t<T>>) and
    requires(T&& t, const scalar_type_of_t<T>& s, I...i) {
      {interface::SetElement<std::decay_t<T>>::set(std::forward<T>(t), s, static_cast<const std::size_t>(i)...)} -> std::same_as<T&&>;
    };
#else
  constexpr bool element_settable = (sizeof...(I) > 0) and (std::is_convertible_v<I, const std::size_t> and ...) and
    (not std::is_const_v<std::remove_reference_t<T>>) and detail::element_settable_impl<T, void, I...>::value;
#endif


  // --------------------------------------- //
  //  has_nested_matrix, nested_matrix_of_t  //
  // --------------------------------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct has_nested_matrix_impl : std::false_type {};

    template<typename T>
    struct has_nested_matrix_impl<T, std::enable_if_t<
      (std::tuple_size<typename interface::Dependencies<T>::type>::value > 0)>> : std::true_type {};
  }
#endif

  /**
   * \brief A matrix that has a nested matrix, if it is a wrapper type.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept has_nested_matrix = (std::tuple_size_v<typename interface::Dependencies<std::decay_t<T>>::type> > 0);
#else
  constexpr bool has_nested_matrix = detail::has_nested_matrix_impl<std::decay_t<T>>::value;
#endif


  /**
   * \brief A wrapper type's nested matrix, if it exists.
   * \details For example, for OpenKalman::Mean<RowCoefficients, M>, the nested matrix type is M.
   * \tparam T A wrapper type that has a nested matrix.
   * \tparam i Index of the dependency (0 by default)
   * \internal \sa interface::Dependencies
   */
#ifdef __cpp_concepts
  template<typename T, std::size_t i = 0> requires
    (i < std::tuple_size_v<typename interface::Dependencies<std::decay_t<T>>::type>)
  using nested_matrix_of = std::tuple_element<i, typename interface::Dependencies<std::decay_t<T>>::type>;
#else
  template<typename T, std::size_t i = 0, typename = void>
  struct nested_matrix_of {};

  template<typename T, std::size_t i>
  struct nested_matrix_of<T, i, std::enable_if_t<
    (i < std::tuple_size<typename interface::Dependencies<std::decay_t<T>>::type>::value)>>
    : std::tuple_element<i, typename interface::Dependencies<std::decay_t<T>>::type> {};
#endif


  /**
   * \brief Helper type for \ref nested_matrix_of.
   * \tparam T A wrapper type that has a nested matrix.
   * \tparam i Index of the dependency (0 by default)
   */
#ifdef __cpp_concepts
  template<typename T, std::size_t i = 0> requires
      (i < std::tuple_size_v<typename interface::Dependencies<std::decay_t<T>>::type>)
  using nested_matrix_of_t = typename nested_matrix_of<T, i>::type;
#else
  template<typename T, std::size_t i = 0>
  using nested_matrix_of_t = typename nested_matrix_of<T, i>::type;
#endif


  // ---------------- //
  //  self_contained  //
  // ---------------- //

  namespace detail
  {
#ifdef __cpp_concepts
    template<typename T>
#else
    template<typename T, typename = void>
#endif
    struct self_contained_impl : std::false_type {};


    template<typename Tup, std::size_t...I>
    constexpr bool no_lvalue_ref_dependencies(std::index_sequence<I...>)
    {
      return (self_contained_impl<std::tuple_element_t<I, Tup>>::value and ...);
    }


#ifdef __cpp_concepts
    template<typename T> requires (not std::is_lvalue_reference_v<T>) and
      (detail::no_lvalue_ref_dependencies<typename interface::Dependencies<std::decay_t<T>>::type>(
        std::make_index_sequence<std::tuple_size_v<typename interface::Dependencies<std::decay_t<T>>::type>> {}))
    struct self_contained_impl<T> : std::true_type {};
#else
    template<typename T>
    struct self_contained_impl<T, std::enable_if_t<
      (std::tuple_size<typename interface::Dependencies<std::decay_t<T>>::type>::value >= 0)>>
      : std::bool_constant<(not std::is_lvalue_reference_v<T>) and
          no_lvalue_ref_dependencies<typename interface::Dependencies<std::decay_t<T>>::type>(
          std::make_index_sequence<std::tuple_size_v<typename interface::Dependencies<std::decay_t<T>>::type>> {})> {};
#endif


    template<typename Tup, std::size_t...I>
    constexpr bool all_lvalue_ref_dependencies_impl(std::index_sequence<I...>)
    {
      return ((sizeof...(I) > 0) and ... and std::is_lvalue_reference_v<std::tuple_element_t<I, Tup>>);
    }


    template<typename T, std::size_t...I>
    constexpr bool no_recursive_runtime_parameters(std::index_sequence<I...>)
    {
      return ((not Dependencies<T>::has_runtime_parameters) and ... and
        no_recursive_runtime_parameters<std::decay_t<std::tuple_element_t<I, typename Dependencies<T>::type>>>(
          std::make_index_sequence<std::tuple_size_v<typename Dependencies<std::decay_t<std::tuple_element_t<I, typename Dependencies<T>::type>>>::type>> {}
          ));
    }

#ifdef __cpp_concepts
    template<typename T>
    concept all_lvalue_ref_dependencies =
      no_recursive_runtime_parameters<std::decay_t<T>>(
        std::make_index_sequence<std::tuple_size_v<typename Dependencies<std::decay_t<T>>::type>> {}) and
      all_lvalue_ref_dependencies_impl<typename Dependencies<std::decay_t<T>>::type>(
        std::make_index_sequence<std::tuple_size_v<typename Dependencies<std::decay_t<T>>::type>> {});
#else
    template<typename T, typename = void>
    struct has_no_runtime_parameters_impl : std::false_type {};

    template<typename T>
    struct has_no_runtime_parameters_impl<T, std::enable_if_t<not Dependencies<T>::has_runtime_parameters>>
      : std::true_type {};


    template<typename T, typename = void>
    struct all_lvalue_ref_dependencies_detail : std::false_type {};

    template<typename T>
    struct all_lvalue_ref_dependencies_detail<T, std::void_t<typename Dependencies<T>::type>>
      : std::bool_constant<has_no_runtime_parameters_impl<T>::value and
        (all_lvalue_ref_dependencies_impl<typename Dependencies<T>::type>(
          std::make_index_sequence<std::tuple_size_v<typename Dependencies<T>::type>> {}))> {};

    template<typename T>
    constexpr bool all_lvalue_ref_dependencies = all_lvalue_ref_dependencies_detail<std::decay_t<T>>::value;


    template<typename T, typename = void>
    struct convert_to_self_contained_is_defined : std::false_type {};

    template<typename T>
    struct convert_to_self_contained_is_defined<T,
      std::void_t<decltype(Dependencies<std::decay_t<T>>::convert_to_self_contained(std::declval<T&&>()))>>
      : std::true_type {};
#endif
  } // namespace detail


  /**
   * \brief Specifies that a type is a self-contained matrix or expression.
   * \details A value is self-contained if it can be created in a function and returned as the result.
   * \sa make_self_contained, equivalent_self_contained_t
   * \internal \sa Dependencies
   */
  template<typename T, typename...Ts>
#ifdef __cpp_concepts
  concept self_contained =
#else
  constexpr bool self_contained =
#endif
    detail::self_contained_impl<T>::value or
    ((sizeof...(Ts) > 0) and ... and (std::is_lvalue_reference_v<Ts> or detail::all_lvalue_ref_dependencies<Ts>));


  // ---------------------- //
  //  constant_coefficient  //
  // ---------------------- //

  /**
   * \brief The constant associated with T, assuming T is a \ref constant_matrix.
   * \details Before using this value, always check if T is a \ref constant_matrix, because
   * the value may be defined in some cases where T is not actually a constant matrix.
   */
#ifdef __cpp_concepts
  template<indexible T>
#else
  template<typename T, typename = void>
#endif
  struct constant_coefficient
  {
    explicit constexpr constant_coefficient(const std::decay_t<T>&) {};
  };


  /**
   * \brief Deduction guide for \ref constant_coefficient.
   */
  template<typename T>
  explicit constant_coefficient(const T&) -> constant_coefficient<T>;


  /**
   * \brief Helper template for constant_coefficient.
   */
#ifdef __cpp_concepts
  template<indexible T>
#else
  template<typename T>
#endif
  constexpr auto constant_coefficient_v = constant_coefficient<T>::value;


  // ------------------------------- //
  //  constant_diagonal_coefficient  //
  // ------------------------------- //

  /**
   * \brief The constant associated with T, assuming T is a \ref constant_diagonal_matrix.
   * \details Before using this value, always check if T is a \ref constant_diagonal_matrix, because
   * the value may be defined in some cases where T is not actually a constant diagonal matrix.
   */
#ifdef __cpp_concepts
  template<indexible T>
#else
  template<typename T, typename = void>
#endif
  struct constant_diagonal_coefficient
  {
    explicit constexpr constant_diagonal_coefficient(const std::decay_t<T>&) {};
  };


  /**
   * \brief Deduction guide for \ref constant_diagonal_coefficient.
   */
  template<typename T>
  explicit constant_diagonal_coefficient(T&&) -> constant_diagonal_coefficient<std::decay_t<T>>;


  /// Helper template for constant_diagonal_coefficient.
#ifdef __cpp_concepts
  template<indexible T>
#else
  template<typename T>
#endif
  constexpr auto constant_diagonal_coefficient_v = constant_diagonal_coefficient<T>::value;


  // ----------------- //
  //  constant_matrix  //
  // ----------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, Likelihood b, typename = void>
    struct scalar_status_is : std::false_type {};

    template<typename T, Likelihood b>
    struct scalar_status_is<T, b, std::enable_if_t<std::decay_t<T>::status == b>> : std::true_type {};
  }
#endif


  /**
   * \brief Specifies that all elements of an object are the same constant value.
   */
  template<typename T, CompileTimeStatus c = CompileTimeStatus::any, Likelihood b = Likelihood::definitely>
#ifdef __cpp_concepts
  concept constant_matrix = indexible<T> and scalar_constant<constant_coefficient<T>, c> and
    (b == Likelihood::maybe or constant_coefficient<T>::status == b);
#else
  constexpr bool constant_matrix = indexible<T> and scalar_constant<constant_coefficient<T>, c> and
    (b == Likelihood::maybe or detail::scalar_status_is<constant_coefficient<T>, b>::value);
#endif


  // -------------------------- //
  //  constant_diagonal_matrix  //
  // -------------------------- //

  /**
   * \brief Specifies that all diagonal elements of a diagonal object are the same constant value.
   * \todo rename to scalar_matrix
   */
  template<typename T, CompileTimeStatus c = CompileTimeStatus::any, Likelihood b = Likelihood::definitely>
#ifdef __cpp_concepts
  concept constant_diagonal_matrix = indexible<T> and scalar_constant<constant_diagonal_coefficient<T>, c> and
    (b == Likelihood::maybe or constant_diagonal_coefficient<T>::status == b);
#else
  constexpr bool constant_diagonal_matrix =
    indexible<T> and scalar_constant<constant_diagonal_coefficient<T>, c> and
    (b == Likelihood::maybe or detail::scalar_status_is<constant_diagonal_coefficient<T>, b>::value);
#endif


  // ------------- //
  //  zero_matrix  //
  // ------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_zero_matrix : std::false_type {};

    template<typename T>
    struct is_zero_matrix<T, std::enable_if_t<constant_matrix<T, CompileTimeStatus::known, Likelihood::maybe>>>
      : std::bool_constant<are_within_tolerance(constant_coefficient_v<T>, 0)> {};
  }
#endif


  /**
   * \brief Specifies that a type is known at compile time to be a constant matrix of value zero.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept zero_matrix = constant_matrix<T, CompileTimeStatus::known, Likelihood::maybe> and
    are_within_tolerance(constant_coefficient_v<T>, 0);
#else
  constexpr bool zero_matrix = detail::is_zero_matrix<T>::value;
#endif


  // ----------------- //
  //  identity_matrix  //
  // ----------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, Likelihood b, typename = void>
    struct is_identity_matrix : std::false_type {};

    template<typename T, Likelihood b>
    struct is_identity_matrix<T, b, std::enable_if_t<constant_diagonal_matrix<T, CompileTimeStatus::known, b>>>
      : std::bool_constant<are_within_tolerance(constant_diagonal_coefficient_v<T>, 1)> {};
  }
#endif

  /**
   * \brief Specifies that a type is an identity matrix.
   */
  template<typename T, Likelihood b = Likelihood::definitely>
#ifdef __cpp_concepts
  concept identity_matrix = constant_diagonal_matrix<T, CompileTimeStatus::known, b> and
    are_within_tolerance(constant_diagonal_coefficient_v<T>, 1);
#else
  constexpr bool identity_matrix = detail::is_identity_matrix<T, b>::value;
#endif


  // ------------------- //
  //  one_by_one_matrix  //
  // ------------------- //

  namespace detail
  {
#ifndef __cpp_concepts
    template<typename T, Likelihood b, typename = void>
    struct has_is_one_by_one_interface : std::false_type {};

    template<typename T, Likelihood b>
    struct has_is_one_by_one_interface<T, b, std::void_t<decltype(interface::IndexTraits<std::decay_t<T>>::template is_one_by_one<b>)>>
      : std::true_type {};


    template<typename T, Likelihood b, typename = void>
    struct is_one_by_one_matrix : std::false_type {};

    template<typename T, Likelihood b>
    struct is_one_by_one_matrix<T, b, std::enable_if_t<interface::IndexTraits<std::decay_t<T>>::template is_one_by_one<b>>>
      : std::true_type {};
#endif

    template<typename T, Likelihood b, std::size_t...I>
    constexpr bool has_1_by_1_dims(std::index_sequence<I...>)
    {
      return (((b != Likelihood::definitely and dynamic_dimension<T, I>) or index_dimension_of_v<T, I> == 1) and ...);
    }
  } // namespace detail


  /**
   * \brief Specifies that a type is a one-by-one matrix (i.e., one row and one column).
   */
  template<typename T, Likelihood b = Likelihood::definitely>
#ifdef __cpp_concepts
  concept one_by_one_matrix = indexible<T> and (max_indices_of_v<T> > 0) and
    (not requires { interface::IndexTraits<std::decay_t<T>>::template is_one_by_one<b>; } or
      interface::IndexTraits<std::decay_t<T>>::template is_one_by_one<b>) and
    (requires { interface::IndexTraits<std::decay_t<T>>::template is_one_by_one<b>; } or
      detail::has_1_by_1_dims<T, b>(std::make_index_sequence<max_indices_of_v<T>>{}));
#else
  constexpr bool one_by_one_matrix = indexible<T> and (max_indices_of_v<T> > 0) and
    (detail::has_is_one_by_one_interface<T, b>::value ? detail::is_one_by_one_matrix<T, b>::value :
      detail::has_1_by_1_dims<T, b>(std::make_index_sequence<max_indices_of_v<T>>{}));
#endif


  // --------------- //
  //  square_matrix  //
  // --------------- //

  namespace detail
  {
    template<typename T>
    constexpr bool maybe_square_matrix_impl(std::index_sequence<>) { return true; }

    template<typename T, std::size_t I0, std::size_t...I>
    constexpr bool maybe_square_matrix_impl(std::index_sequence<I0, I...>)
    {
      if constexpr (dynamic_dimension<T, I0>)
        return maybe_square_matrix_impl<T>(std::index_sequence<I...>{});
      else
        return ((dynamic_dimension<T, I> or equivalent_to<coefficient_types_of_t<T, I0>, coefficient_types_of_t<T, I>>) and ...);
    }


    template<typename T>
    struct is_maybe_square_matrix
      : std::bool_constant<maybe_square_matrix_impl<T>(std::make_index_sequence<max_indices_of_v<T>>{})> {};

#ifndef __cpp_concepts
    template<typename T, Likelihood b, typename = void>
    struct has_is_square_interface : std::false_type {};

    template<typename T, Likelihood b>
    struct has_is_square_interface<T, b, std::void_t<decltype(interface::IndexTraits<std::decay_t<T>>::template is_square<b>)>>
      : std::true_type {};


    template<typename T, Likelihood b, typename = void>
    struct is_explicitly_square : std::false_type {};

    template<typename T, Likelihood b>
    struct is_explicitly_square<T, b, std::enable_if_t<interface::IndexTraits<std::decay_t<T>>::template is_square<b>>>
      : std::true_type {};


    template<typename T, typename = void>
    struct is_diagonal_adapter : std::false_type {};

    template<typename T>
    struct is_diagonal_adapter<T, std::enable_if_t<interface::TriangularTraits<std::decay_t<T>>::is_diagonal_adapter>>
      : std::true_type {};


    template<typename T, Likelihood b, typename = void>
    struct is_square_matrix : std::false_type {};

    template<typename T, Likelihood b>
    struct is_square_matrix<T, b, std::enable_if_t<indexible<T> and (max_indices_of<T>::value > 0)>> : std::bool_constant<
      (b != Likelihood::definitely or not has_dynamic_dimensions<T>) and
      (max_indices_of_v<T> != 1 or dynamic_dimension<T, 0> or equivalent_to<coefficient_types_of_t<T, 0>, Axis>) and
      (max_indices_of_v<T> < 2 or is_maybe_square_matrix<T>::value)> {};


    template<typename T, TriangleType t, Likelihood b, typename = void>
    struct is_triangular_matrix : std::false_type {};

    template<typename T, TriangleType t, Likelihood b>
    struct is_triangular_matrix<T, t, b, std::enable_if_t<interface::TriangularTraits<std::decay_t<T>>::template is_triangular<t, b>>>
      : std::true_type {};
#endif
  } // namespace detail


  /**
   * \brief Specifies that a matrix is square (i.e., has equivalent index descriptors along each dimension).
   */
  template<typename T, Likelihood b = Likelihood::definitely>
#ifdef __cpp_concepts
  concept square_matrix = one_by_one_matrix<T, b> or (indexible<T> and (max_indices_of_v<T> > 0) and
    (not requires { interface::IndexTraits<std::decay_t<T>>::template is_square<b>; } or
      interface::IndexTraits<std::decay_t<T>>::template is_square<b>) and
    (requires { interface::IndexTraits<std::decay_t<T>>::template is_square<b>; } or
      ((b != Likelihood::definitely or not has_dynamic_dimensions<T>) and
        (max_indices_of_v<T> != 1 or dynamic_dimension<T, 0> or equivalent_to<coefficient_types_of_t<T, 0>, Axis>) and
        (max_indices_of_v<T> < 2 or detail::is_maybe_square_matrix<T>::value)) or
      (b == Likelihood::definitely and
        (interface::TriangularTraits<std::decay_t<T>>::template is_triangular<TriangleType::any, b> or
          interface::TriangularTraits<std::decay_t<T>>::is_diagonal_adapter))));
#else
  constexpr bool square_matrix = one_by_one_matrix<T, b> or
    ((not detail::has_is_square_interface<T, b>::value or detail::is_explicitly_square<T, b>::value) and
    (detail::has_is_square_interface<T, b>::value or detail::is_square_matrix<T, b>::value or
      (b == Likelihood::definitely and (detail::is_triangular_matrix<std::decay_t<T>, TriangleType::any, b>::value or
         detail::is_diagonal_adapter<T>::value))));
#endif


  // ------------------- //
  //  triangular_matrix  //
  // ------------------- //

  /**
   * \brief Specifies that a type is a triangular matrix (upper, lower, or diagonal).
   * \tparam T A matrix or tensor.
   */
  template<typename T, TriangleType t = TriangleType::any, Likelihood b = Likelihood::definitely>
#ifdef __cpp_concepts
  concept triangular_matrix = indexible<T> and
    ((interface::TriangularTraits<std::decay_t<T>>::template is_triangular<t, square_matrix<T> ? Likelihood::maybe : b> and square_matrix<T, b>) or
    interface::TriangularTraits<std::decay_t<T>>::is_diagonal_adapter or constant_diagonal_matrix<T, CompileTimeStatus::any, b>);
#else
  constexpr bool triangular_matrix =
    ((detail::is_triangular_matrix<T, t, square_matrix<T> ? Likelihood::maybe : b>::value and square_matrix<T, b>) or
    detail::is_diagonal_adapter<T>::value or constant_diagonal_matrix<T, CompileTimeStatus::any, b>);
#endif


  // ------------------ //
  //  triangle_type_of  //
  // ------------------ //

  /**
   * \brief The TriangleType associated with a matrix, or the common TriangleType associated with a set of matrices.
   * \details If there is no common triangle type, the result is TriangleType::any.
   */
  template<typename T, typename...Ts>
  struct triangle_type_of
    : std::integral_constant<TriangleType,
      (triangular_matrix<T, TriangleType::diagonal, Likelihood::maybe> and ... and triangular_matrix<Ts, TriangleType::diagonal, Likelihood::maybe>) ? TriangleType::diagonal :
      (triangular_matrix<T, TriangleType::lower, Likelihood::maybe> and ... and triangular_matrix<Ts, TriangleType::lower, Likelihood::maybe>) ? TriangleType::lower :
      (triangular_matrix<T, TriangleType::upper, Likelihood::maybe> and ... and triangular_matrix<Ts, TriangleType::upper, Likelihood::maybe>) ? TriangleType::upper :
      TriangleType::any> {};


  /**
   * \brief The TriangleType associated with a \ref triangular_matrix.
   */
  template<typename T, typename...Ts>
  constexpr auto triangle_type_of_v = triangle_type_of<T, Ts...>::value;


  // -------------------- //
  //  triangular_adapter  //
  // -------------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_triangular_adapter : std::false_type {};

    template<typename T>
    struct is_triangular_adapter<T, std::enable_if_t<interface::TriangularTraits<std::decay_t<T>>::is_triangular_adapter>>
      : std::true_type {};
  }
#endif


  /**
   * \brief Specifies that a type is a triangular adapter of triangle type triangle_type.
   * \details If T has a dynamic shape, it is not guaranteed to be triangular because it could be non-square.
   * \tparam T A matrix or tensor.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept triangular_adapter = interface::TriangularTraits<std::decay_t<T>>::is_triangular_adapter and
#else
  constexpr bool triangular_adapter = detail::is_triangular_adapter<T>::value and has_nested_matrix<T> and
#endif
    has_nested_matrix<T> and square_matrix<T, Likelihood::maybe>;


  // ----------------- //
  //  diagonal_matrix  //
  // ----------------- //

  /**
   * \brief Specifies that a type is a diagonal matrix.
   */
  template<typename T, Likelihood b = Likelihood::definitely>
#ifdef __cpp_concepts
  concept diagonal_matrix =
#else
  constexpr bool diagonal_matrix =
#endif
    triangular_matrix<T, TriangleType::diagonal, b>;


  // ------------------ //
  //  diagonal_adapter  //
  // ------------------ //

  /**
   * \brief Specifies that a type is a diagonal adapter.
   * \details This is a wrapper that takes elements of a matrix or tensor and distributes them along a diagonal.
   * The rest of the elements are zero.
   * \tparam T A matrix or tensor.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept diagonal_adapter = indexible<T> and interface::TriangularTraits<std::decay_t<T>>::is_diagonal_adapter and
    has_nested_matrix<T> and (max_indices_of_v<T> >= 2);
#else
  constexpr bool diagonal_adapter = indexible<T> and detail::is_diagonal_adapter<T>::value and has_nested_matrix<T> and
    (max_indices_of_v<T> >= 2);
#endif


  // ------------------ //
  //  hermitian_matrix  //
  // ------------------ //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_hermitian_matrix : std::false_type {};

    template<typename T>
    struct is_hermitian_matrix<T, std::enable_if_t<interface::HermitianTraits<std::decay_t<T>>::is_hermitian>>
      : std::true_type {};


    template<typename T, Likelihood b, typename = void>
    struct is_inferred_hermitian_matrix : std::false_type {};

    template<typename T, Likelihood b>
    struct is_inferred_hermitian_matrix<T, b, std::enable_if_t<not complex_number<typename scalar_type_of<T>::type> or
      zero_matrix<T> or real_axis_number<constant_coefficient<T>> or real_axis_number<constant_diagonal_coefficient<T>>>>
      : std::true_type {};
  };
#endif


  /**
   * \brief Specifies that a type is a hermitian matrix.
   * \T A matrix or tensor.
   */
  template<typename T, Likelihood b = Likelihood::definitely>
#ifdef __cpp_concepts
  concept hermitian_matrix = indexible<T> and
    ((interface::HermitianTraits<std::decay_t<T>>::is_hermitian and square_matrix<T, b>) or
      (((constant_matrix<T, CompileTimeStatus::any, b> and square_matrix<T, b>) or diagonal_matrix<T, b>) and
      (not complex_number<scalar_type_of_t<T>> or zero_matrix<T> or
          real_axis_number<constant_coefficient<T>> or real_axis_number<constant_diagonal_coefficient<T>>)));
#else
  constexpr bool hermitian_matrix = (detail::is_hermitian_matrix<T>::value and square_matrix<T, b>) or
    (((constant_matrix<T, CompileTimeStatus::any, b> and square_matrix<T, b>) or diagonal_matrix<T, b>) and
      detail::is_inferred_hermitian_matrix<T, b>::value);
#endif


  // ------------------- //
  //  hermitian_adapter  //
  // ------------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, HermitianAdapterType t, typename = void>
    struct is_hermitian_adapter : std::false_type {};

    template<typename T, HermitianAdapterType t>
    struct is_hermitian_adapter<T, t, std::enable_if_t<
        std::is_convertible_v<decltype(interface::HermitianTraits<std::decay_t<T>>::adapter_type), const HermitianAdapterType>>>
      : std::bool_constant<(interface::HermitianTraits<std::decay_t<T>>::adapter_type == HermitianAdapterType::lower or
          interface::HermitianTraits<std::decay_t<T>>::adapter_type == HermitianAdapterType::upper) and
        (t == HermitianAdapterType::any or interface::HermitianTraits<std::decay_t<T>>::adapter_type == t)> {};
  };
#endif


  /**
   * \brief Specifies that a type is a hermitian matrix adapter of a particular type.
   * \details If T has a dynamic shape, it is not guaranteed to be hermitian because it could be non-square.
   * \tparam T A matrix or tensor.
   * \tparam t The HermitianAdapterType of T.
   */
  template<typename T, HermitianAdapterType t = HermitianAdapterType::any>
#ifdef __cpp_concepts
  concept hermitian_adapter = indexible<T> and has_nested_matrix<T> and
    (interface::HermitianTraits<std::decay_t<T>>::adapter_type == HermitianAdapterType::lower or
      interface::HermitianTraits<std::decay_t<T>>::adapter_type == HermitianAdapterType::upper) and
    (t == HermitianAdapterType::any or interface::HermitianTraits<std::decay_t<T>>::adapter_type == t) and
    square_matrix<T, Likelihood::maybe>;
#else
  constexpr bool hermitian_adapter = indexible<T> and has_nested_matrix<T> and
    detail::is_hermitian_adapter<std::decay_t<T>, t>::value and square_matrix<T, Likelihood::maybe>;
#endif


  // --------------------------- //
  //  hermitian_adapter_type_of  //
  // --------------------------- //

  /**
   * \brief The TriangleType associated with the storage triangle of one or more matrices.
   * \details If there is no common triangle type, the result is TriangleType::any.
   * If the matrices have a dynamic shape, the result assumes the matrices are square.
   */
  template<typename T, typename...Ts>
  struct hermitian_adapter_type_of : std::integral_constant<HermitianAdapterType,
    (hermitian_adapter<T, HermitianAdapterType::lower> and ... and hermitian_adapter<Ts, HermitianAdapterType::lower>) ? HermitianAdapterType::lower :
    (hermitian_adapter<T, HermitianAdapterType::upper> and ... and hermitian_adapter<Ts, HermitianAdapterType::upper>) ? HermitianAdapterType::upper :
    HermitianAdapterType::any> {};


  /**
   * \brief The TriangleType associated with the storage triangle of a \ref hermitian_matrix.
   * \details Possible values are \ref HermitianAdapterType::lower "lower", \ref HermitianAdapterType::upper "upper", or
   * \ref HermitianAdapterType::any "any".
   */
  template<typename T, typename...Ts>
  constexpr auto hermitian_adapter_type_of_v = hermitian_adapter_type_of<T, Ts...>::value;


  // --------------------------------------------------------- //
  //   dimension_size_of_index_is, row_vector, column_vector   //
  // --------------------------------------------------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, std::size_t index, std::size_t value, typename = void>
    struct dimension_size_of_index_is_impl : std::false_type {};

    template<typename T, std::size_t index, std::size_t value>
    struct dimension_size_of_index_is_impl<T, index, value, std::enable_if_t<
      index_dimension_of<T, index>::value == value>> : std::true_type {};
  }
#endif


  /**
   * \brief Specifies that a given index of T has a specified size.
   */
  template<typename T, std::size_t index, std::size_t value, Likelihood b = Likelihood::definitely>
#ifdef __cpp_concepts
  concept dimension_size_of_index_is = (index_dimension_of_v<T, index> == value) or
#else
  constexpr bool dimension_size_of_index_is = detail::dimension_size_of_index_is_impl<T, index, value>::value or
#endif
    (b == Likelihood::maybe and dynamic_dimension<T, index>);


  /**
   * \brief Specifies that T is a row vector.
   * \todo Remove as redundant?
   */
  template<typename T, Likelihood b = Likelihood::definitely>
#ifdef __cpp_concepts
  concept row_vector =
#else
  constexpr bool row_vector =
#endif
    dimension_size_of_index_is<T, 0, 1, b>;


  /**
   * \brief Specifies that T is a column vector.
   * \todo Remove as redundant?
   */
  template<typename T, Likelihood b = Likelihood::definitely>
#ifdef __cpp_concepts
  concept column_vector =
#else
  constexpr bool column_vector =
#endif
    dimension_size_of_index_is<T, 1, 1, b>;


  // ------------------------- //
  //  maybe_has_same_shape_as  //
  // ------------------------- //

  namespace detail
  {
    template<std::size_t I>
    constexpr bool maybe_dimensions_are_same() { return true; }

    template<std::size_t I, typename T, typename...Ts>
    constexpr bool maybe_dimensions_are_same()
    {
      if constexpr (dynamic_dimension<T, I>) return maybe_dimensions_are_same<I, Ts...>();
      else return ((dynamic_dimension<Ts, I> or equivalent_to<coefficient_types_of_t<T, I>, coefficient_types_of_t<Ts, I>>) and ...);
    }

    template<typename...Ts, std::size_t...Is>
    constexpr bool maybe_has_same_shape_as_impl(std::index_sequence<Is...>)
    {
      return (maybe_dimensions_are_same<Is, Ts...>() and ...);
    }
#ifndef __cpp_concepts


    template<typename = void, typename...Ts>
    struct maybe_has_same_shape_as_test : std::false_type {};

    template<typename...Ts>
    struct maybe_has_same_shape_as_test<std::enable_if_t<(indexible<Ts> and ...)>, Ts...> : std::true_type {};
#endif
  }

  /**
   * \brief Specifies that it is not ruled out, at compile time, that T has the same shape as Ts.
   * \details Two dimensions are considered the same if their index descriptors are \ref equivalent_to "equivalent".
   * \sa has_same_shape_as
   */
  template<typename...Ts>
#ifdef __cpp_concepts
  concept maybe_has_same_shape_as = (indexible<Ts> and ...) and
    detail::maybe_has_same_shape_as_impl<Ts...>(std::make_index_sequence<std::max({std::size_t{0}, max_indices_of_v<Ts>...})>{});
#else
  constexpr bool maybe_has_same_shape_as = detail::maybe_has_same_shape_as_test<void, Ts...>::value;
#endif


  // ------------------- //
  //  has_same_shape_as  //
  // ------------------- //

  /**
   * \brief Specifies that T has the same shape as Ts.
   * \details Two dimensions are considered the same if their index descriptors are \ref equivalent_to "equivalent".
   * \sa maybe_has_same_shape_as
   */
  template<typename...Ts>
#ifdef __cpp_concepts
  concept has_same_shape_as =
#else
  constexpr bool has_same_shape_as =
#endif
    maybe_has_same_shape_as<Ts...> and ((not has_dynamic_dimensions<Ts>) and ...);


  // ---------------------- //
  //   mean, wrapped_mean   //
  // ---------------------- //

  namespace internal
  {
    template<typename T>
    struct is_mean : std::false_type {};
  }


  /**
   * \brief Specifies that T is a mean (i.e., is a specialization of the class Mean).
   */
  template<typename T>
#ifdef __cpp_concepts
  concept mean = internal::is_mean<std::decay_t<T>>::value;
#else
  constexpr bool mean = internal::is_mean<std::decay_t<T>>::value;
#endif


  /**
   * \brief Specifies that T is a wrapped mean (i.e., its row fixed_index_descriptor have at least one type that requires wrapping).
   */
#ifdef __cpp_concepts
  template<typename T>
  concept wrapped_mean =
#else
  template<typename T>
  constexpr bool wrapped_mean =
#endif
    mean<T> and (not has_untyped_index<T, 0>);


  // ----------------------------------------- //
  //   euclidean_mean, euclidean_transformed   //
  // ----------------------------------------- //

  namespace internal
  {
    template<typename T>
    struct is_euclidean_mean : std::false_type {};
  }


  /**
   * \brief Specifies that T is a Euclidean mean (i.e., is a specialization of the class EuclideanMean).
   */
  template<typename T>
#ifdef __cpp_concepts
  concept euclidean_mean = internal::is_euclidean_mean<std::decay_t<T>>::value;
#else
  constexpr bool euclidean_mean = internal::is_euclidean_mean<std::decay_t<T>>::value;
#endif


  /**
   * \brief Specifies that T is a Euclidean mean that actually has coefficients that are transformed to Euclidean space.
   */
#ifdef __cpp_concepts
  template<typename T>
  concept euclidean_transformed =
#else
  template<typename T>
  constexpr bool euclidean_transformed =
#endif
    euclidean_mean<T> and (not has_untyped_index<T, 0>);


  // ---------------- //
  //   typed matrix   //
  // ---------------- //

  namespace internal
  {
    template<typename T>
    struct is_matrix : std::false_type {};
  }


  /**
   * \brief Specifies that T is a typed matrix (i.e., is a specialization of Matrix, Mean, or EuclideanMean).
   */
  template<typename T>
#ifdef __cpp_concepts
  concept typed_matrix = mean<T> or euclidean_mean<T> or internal::is_matrix<std::decay_t<T>>::value;
#else
  constexpr bool typed_matrix = mean<T> or euclidean_mean<T> or internal::is_matrix<std::decay_t<T>>::value;
#endif


  // ------------------------------------------------------------- //
  //  covariance, self_adjoint_covariance, triangular_covariance  //
  // ------------------------------------------------------------- //

  namespace internal
  {
    template<typename T>
    struct is_self_adjoint_covariance : std::false_type {};
  }


  /**
   * \brief T is a self-adjoint covariance matrix (i.e., a specialization of Covariance).
   */
  template<typename T>
#ifdef __cpp_concepts
  concept self_adjoint_covariance = internal::is_self_adjoint_covariance<std::decay_t<T>>::value;
#else
  constexpr bool self_adjoint_covariance = internal::is_self_adjoint_covariance<std::decay_t<T>>::value;
#endif


  namespace internal
  {
    template<typename T>
    struct is_triangular_covariance : std::false_type {};
  }


  /**
   * \brief T is a square root (Cholesky) covariance matrix (i.e., a specialization of SquareRootCovariance).
   */
  template<typename T>
#ifdef __cpp_concepts
  concept triangular_covariance = internal::is_triangular_covariance<std::decay_t<T>>::value;
#else
  constexpr bool triangular_covariance = internal::is_triangular_covariance<std::decay_t<T>>::value;
#endif


  /**
   * \brief T is a specialization of either Covariance or SquareRootCovariance.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept covariance = self_adjoint_covariance<T> or triangular_covariance<T>;
#else
  constexpr bool covariance = self_adjoint_covariance<T> or triangular_covariance<T>;
#endif


  // --------------- //
  //  distributions  //
  // --------------- //

  namespace internal
  {
    template<typename T>
    struct is_gaussian_distribution : std::false_type {};
  }

  /**
   * \brief T is a Gaussian distribution.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept gaussian_distribution = internal::is_gaussian_distribution<std::decay_t<T>>::value;
#else
  constexpr bool gaussian_distribution = internal::is_gaussian_distribution<std::decay_t<T>>::value;
#endif


  /**
   * \brief T is a statistical distribution of any kind that is defined in OpenKalman.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept distribution = gaussian_distribution<T>;
#else
  constexpr bool distribution = gaussian_distribution<T>;
#endif


  // --------------- //
  //  cholesky_form  //
  // --------------- //

  namespace detail
  {
#ifndef __cpp_concepts
    template<typename T, typename = void>
    struct is_cholesky_form : std::false_type {};

    template<typename T>
    struct is_cholesky_form<T, std::enable_if_t<covariance<T>>>
      : std::bool_constant<not hermitian_matrix<nested_matrix_of_t<T>>> {};

    template<typename T>
    struct is_cholesky_form<T, std::enable_if_t<distribution<T>>>
      : is_cholesky_form<typename DistributionTraits<T>::Covariance> {};
#endif
  }


  /**
   * \brief Specifies that a type has a nested native matrix that is a Cholesky square root.
   * \details If this is true, then nested_matrix_of_t<T> is true.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept cholesky_form = (not covariance<T> or not hermitian_matrix<nested_matrix_of_t<T>>) or
    (not distribution<T> or not hermitian_matrix<nested_matrix_of_t<typename DistributionTraits<T>::Covariance>>);
#else
  constexpr bool cholesky_form = detail::is_cholesky_form<std::decay_t<T>>::value;
#endif


  // ------------------------- //
  //    covariance_nestable    //
  // ------------------------- //

  /**
   * \brief T is an acceptable nested matrix for a covariance (including triangular_covariance).
   */
  template<typename T>
#ifdef __cpp_concepts
  concept covariance_nestable =
#else
  constexpr bool covariance_nestable =
#endif
    triangular_matrix<T> or hermitian_matrix<T>;


  // --------------------------- //
  //    typed_matrix_nestable    //
  // --------------------------- //

  /**
   * \brief Specifies a type that is nestable in a general typed matrix (e.g., matrix, mean, or euclidean_mean)
   */
  template<typename T>
#ifdef __cpp_concepts
  concept typed_matrix_nestable =
#else
  constexpr bool typed_matrix_nestable =
#endif
    indexible<T>;


  // ---------- //
  //  writable  //
  // ---------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct writable_impl : std::false_type {};

    template<typename T>
    struct writable_impl<T, std::enable_if_t<
      interface::EquivalentDenseWritableMatrix<std::decay_t<T>, scalar_type_of_t<T>>::is_writable>> : std::true_type {};
  };
#endif


  /**
   * \internal
   * \brief Specifies that T is a dense, writable matrix.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept writable =
    indexible<T> and interface::EquivalentDenseWritableMatrix<std::decay_t<T>, scalar_type_of_t<T>>::is_writable and
      (not std::is_const_v<std::remove_reference_t<T>>);
#else
  constexpr bool writable =
    indexible<T> and detail::writable_impl<T>::value and (not std::is_const_v<std::remove_reference_t<T>>);
#endif


  // ------------ //
  //  modifiable  //
  // ------------ //

  namespace internal
  {
#ifdef __cpp_concepts
    template<typename T, typename U>
#else
    template<typename T, typename U, typename = void>
#endif
    struct is_modifiable : std::true_type {};


    // Custom modifiability parameter that can be defined in the native matrix ecosystem.
#ifdef __cpp_concepts
    template<typename T, typename U>
#else
    template<typename T, typename U, typename = void>
#endif
    struct is_modifiable_native : std::true_type {};

  } // namespace internal


  /**
   * \internal
   * \brief Specifies that U is not obviously incompatible with T, such that assigning U to T might be possible.
   * \details The result is true unless there is an incompatibility of some kind that would prevent assignment.
   * Examples of such incompatibility are if T is constant or has a nested constant type, if T and U have a
   * different shape or scalar type, or if T and U differ as to being self-adjoint, triangular, diagonal,
   * zero, or identity. Even if this concept is true, a compile-time error is still possible.
   */
  template<typename T, typename U>
#ifdef __cpp_concepts
  concept modifiable =
#else
  constexpr bool modifiable =
#endif
    indexible<T> and indexible<U> and internal::is_modifiable<T, U>::value and internal::is_modifiable_native<T, U>::value;


} // namespace OpenKalman

#endif //OPENKALMAN_FORWARD_TRAITS_HPP
