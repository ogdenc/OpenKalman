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


  // --------------------- //
  //  max_tensor_order_of  //
  // --------------------- //

  namespace detail
  {
    template<std::size_t max, typename T>
    constexpr std::size_t max_tensor_order_of_impl()
    {
      if constexpr (max == 0)
        return 0;
      else if constexpr (interface::IndexTraits<std::decay_t<T>, max - 1>::dimension == 0)
        return 0;
      else if constexpr (interface::IndexTraits<std::decay_t<T>, max - 1>::dimension == 1)
        return max_tensor_order_of_impl<max - 1, T>();
      else
        return 1 + max_tensor_order_of_impl<max - 1, T>();
    }
  }

  /**
   * \brief The maximum number of indices of structure T.
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

#ifndef __cpp_lib_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_indexible : std::false_type {};

    template<typename T>
    struct is_indexible<T, std::enable_if_t<(max_indices_of<T>::value > 0) and scalar_type<typename scalar_type_of<T>::type>>>
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
  concept indexible = (max_indices_of_v<T> > 0) and scalar_type<scalar_type_of_t<T>>;
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
  template<typename T, std::size_t N = 0>
#else
  template<typename T, std::size_t N = 0, typename = void>
#endif
  struct index_dimension_of;


#ifdef __cpp_concepts
  template<indexible T, std::size_t N> requires (N < max_indices_of_v<T>)
  struct index_dimension_of<T, N>
#else
  template<typename T, std::size_t N>
  struct index_dimension_of<T, N, std::enable_if_t<indexible<T> and N < max_indices_of<T>::value>>
#endif
    : std::integral_constant<std::size_t, interface::IndexTraits<std::decay_t<T>, N>::dimension> {};


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
    constexpr bool number_of_dynamic_indices_impl(std::index_sequence<I...>)
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
    requires { typename interface::CoordinateSystemTraits<std::decay_t<T>, N>::coordinate_system_types; }
  struct coefficient_types_of<T, N>
#else
  template<typename T, std::size_t N>
  struct coefficient_types_of<T, N, std::enable_if_t<indexible<T> and N < max_indices_of_v<T> and
    std::is_void<std::void_t<typename interface::CoordinateSystemTraits<std::decay_t<T>, N>::coordinate_system_types>>::value>>
#endif
  {
    using type = typename interface::CoordinateSystemTraits<std::decay_t<T>, N>::coordinate_system_types;
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

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void, typename...I>
    struct element_gettable_impl : std::false_type {};

    template<typename T, typename...I>
    struct element_gettable_impl<T, std::enable_if_t<(sizeof...(I) <= max_indices_of_v<T>) and
      (std::is_convertible_v<I, const std::size_t&> and ...) and
      std::is_void<std::void_t<decltype(interface::GetElement<std::decay_t<T>, void, I...>::get(
        std::declval<T&&>(), std::declval<I>()...))>>::value and (sizeof...(I) > 0)>, I...>
      : std::true_type {};
  }
#endif


  /**
   * \brief Specifies that a type has elements that can be retrieved with indices I... (of type std::size_t).
   */
  template<typename T, typename...I>
#ifdef __cpp_concepts
  concept element_gettable = (sizeof...(I) <= max_indices_of_v<T>) and
    (std::convertible_to<I, const std::size_t&> and ...) and (sizeof...(I) > 0) and
    requires(T&& t, I...i) { interface::GetElement<std::decay_t<T>, I...>::get(std::forward<T>(t), i...); };
#else
  constexpr bool element_gettable = detail::element_gettable_impl<T, void, I...>::value;
#endif


  // ------------------ //
  //  element_settable  //
  // ------------------ //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void, typename...I>
    struct element_settable_impl : std::false_type {};

    template<typename T, typename...I>
    struct element_settable_impl<T, std::enable_if_t<(sizeof...(I) <= max_indices_of_v<T>) and
      (std::is_convertible_v<I, const std::size_t&> and ...) and
      (not std::is_const_v<std::remove_reference_t<T>>) and
      std::is_void<std::void_t<decltype(interface::SetElement<std::decay_t<T>, void, I...>::set(
        std::declval<std::remove_reference_t<T>&>(),
        std::declval<const scalar_type_of_t<T>&>(), std::declval<I>()...))>>::value and (sizeof...(I) > 0)>, I...>
      : std::true_type {};
  }
#endif


  /**
   * \brief Specifies that a type has elements that can be set with indices I... (of type std::size_t).
   */
  template<typename T, typename...I>
#ifdef __cpp_concepts
  concept element_settable = (sizeof...(I) <= max_indices_of_v<T>) and
    (std::convertible_to<I, const std::size_t&> and ...) and
    (not std::is_const_v<std::remove_reference_t<T>>) and
    requires(std::remove_reference_t<T>& t, const scalar_type_of_t<T>& s, I...i) {
      interface::SetElement<std::decay_t<T>, I...>::set(t, s, i...);
    };
#else
  constexpr bool element_settable = detail::element_settable_impl<T, void, I...>::value;
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


  // ----------------- //
  //  constant_matrix  //
  // ----------------- //

  namespace detail
  {
#ifndef __cpp_concepts
    template<typename T, typename = void>
    struct get_constant_res {};

    template<typename T>
    struct get_constant_res<T, std::void_t<decltype(interface::SingleConstant{std::declval<const std::decay_t<T>&>()}.get_constant())>>
    {
      using type = std::decay_t<decltype(interface::SingleConstant{std::declval<const std::decay_t<T>&>()}.get_constant())>;
    };


    template<typename T, CompileTimeStatus c = CompileTimeStatus::known, typename = void>
    struct is_constant_matrix : std::false_type {};

    template<typename T, CompileTimeStatus c>
    struct is_constant_matrix<T, c, std::enable_if_t<scalar_constant<typename get_constant_res<T>::type, c>>>
      : std::true_type {};


    template<typename T, std::size_t v, typename = void>
    struct is_specific_constant_matrix : std::false_type {};

    template<typename T, std::size_t v>
    struct is_specific_constant_matrix<T, v, std::enable_if_t<are_within_tolerance(get_constant_res<T>::type::value, v)>>
      : std::true_type {};


    template<typename T, Likelihood b, typename = void>
    struct constant_status : std::false_type {};

    template<typename T, Likelihood b>
    struct constant_status<T, b, std::enable_if_t<std::is_convertible_v<decltype(get_constant_res<T>::type::status), Likelihood>>>
      : std::bool_constant<get_constant_res<T>::type::status == b> {};


    template<typename T, typename = void>
    struct get_constant_diagonal_res {};

    template<typename T>
    struct get_constant_diagonal_res<T, std::void_t<decltype(interface::SingleConstant{std::declval<const std::decay_t<T>&>()}.get_constant_diagonal())>>
    {
      using type = std::decay_t<decltype(interface::SingleConstant{std::declval<const std::decay_t<T>&>()}.get_constant_diagonal())>;
    };


    template<typename T, CompileTimeStatus c = CompileTimeStatus::known, typename = void>
    struct is_constant_diagonal_matrix : std::false_type {};

    template<typename T, CompileTimeStatus c>
    struct is_constant_diagonal_matrix<T, c, std::enable_if_t<scalar_constant<typename get_constant_diagonal_res<T>::type, c>>>
      : std::true_type {};


    template<typename T, std::size_t v, typename = void>
    struct is_specific_constant_diagonal_matrix : std::false_type {};

    template<typename T, std::size_t v>
    struct is_specific_constant_diagonal_matrix<T, v, std::enable_if_t<are_within_tolerance(get_constant_diagonal_res<T>::type::value, v)>>
      : std::true_type {};


    template<typename T, Likelihood b, typename = void>
    struct constant_diagonal_status : std::false_type {};

    template<typename T, Likelihood b>
    struct constant_diagonal_status<T, b, std::enable_if_t<
      std::is_convertible_v<decltype(get_constant_diagonal_res<T>::type::status), Likelihood>>>
      : std::bool_constant<get_constant_diagonal_res<T>::type::status == b> {};


    template<typename T, std::size_t I, typename = void>
    struct dimension_is_1D : std::false_type {};

    template<typename T, std::size_t I>
    struct dimension_is_1D<T, I, std::enable_if_t<index_dimension_of<T, I>::value == 1>> : std::true_type {};

#endif
    template<typename T>
    constexpr bool maybe_one_by_one_matrix_impl(std::index_sequence<>) { return true; }

    template<typename T, std::size_t I0, std::size_t...I>
    constexpr bool maybe_one_by_one_matrix_impl(std::index_sequence<I0, I...>)
    {
      if constexpr (dynamic_dimension<T, I0>)
        return maybe_one_by_one_matrix_impl<T>(std::index_sequence<I...>{});
# ifdef __cpp_concepts
      else if constexpr (index_dimension_of_v<T, I0> != 1)
# else
        else if constexpr (not dimension_is_1D<T, I0>::value)
# endif
        return false;
      else
        return ((dynamic_dimension<T, I> or equivalent_to<coefficient_types_of_t<T, I0>, coefficient_types_of_t<T, I>>) and ...);
    }


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

  } // namespace detail


  /**
   * \brief Specifies that a type is a constant matrix, with the constant known at compile time.
   */
  template<typename T, Likelihood b = Likelihood::definitely, CompileTimeStatus c = CompileTimeStatus::known>
#ifdef __cpp_concepts
  concept constant_matrix = indexible<T> and
    requires(interface::SingleConstant<std::decay_t<T>>& trait) { requires
      requires {
        {trait.get_constant()} -> scalar_constant<c>;
        requires b != Likelihood::definitely or
          not requires { requires std::decay_t<decltype(trait.get_constant())>::status == Likelihood::maybe; };
      } or
      requires {
        {trait.get_constant_diagonal()} -> scalar_constant<c>;
        requires requires { requires are_within_tolerance(std::decay_t<decltype(trait.get_constant_diagonal())>::value, 0); } or
          (detail::maybe_one_by_one_matrix_impl<T>(std::make_index_sequence<max_indices_of_v<T>>{}) and
            (b != Likelihood::definitely or not has_dynamic_dimensions<T>));
      };
    };
#else
  constexpr bool constant_matrix = indexible<T> and (
    (detail::is_constant_matrix<T, c>::value and
      (b != Likelihood::definitely or not detail::constant_status<T, Likelihood::maybe>::value)) or
    (detail::is_constant_diagonal_matrix<T, c>::value and
      (detail::is_specific_constant_diagonal_matrix<T, 0>::value or
      (detail::maybe_one_by_one_matrix_impl<T>(std::make_index_sequence<max_indices_of_v<T>>{}) and
        (b != Likelihood::definitely or not has_dynamic_dimensions<T>)))));
#endif


  // ---------------------- //
  //  constant_coefficient  //
  // ---------------------- //

  /**
   * \brief The constant associated with T, assuming T is a \ref constant_matrix.
   * \details Before using this value, always check if T is a \ref constant_matrix, because the value be defined exist
   * even in cases where T is not actually a constant matrix.
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
   * \overload
   * \brief In this case, the constant can be derived at compile time.
   */
#ifdef __cpp_concepts
  template<constant_matrix<Likelihood::maybe, CompileTimeStatus::known> T>
  struct constant_coefficient<T>
#else
  template<typename T>
  struct constant_coefficient<T, std::enable_if_t<constant_matrix<T, Likelihood::maybe, CompileTimeStatus::known>>>
#endif
  {
    constexpr constant_coefficient() = default;
    explicit constexpr constant_coefficient(const std::decay_t<T>&) {};
    using value_type = scalar_type_of_t<T>;
    using type = constant_coefficient;

    static constexpr value_type value =
      []{
        using Trait = interface::SingleConstant<std::decay_t<T>>;
#ifdef __cpp_concepts
        if constexpr (requires(Trait& trait) { {trait.get_constant()} -> scalar_constant<CompileTimeStatus::known>; })
#else
        if constexpr (detail::is_constant_matrix<T>::value)
#endif
          return std::decay_t<decltype(std::declval<Trait>().get_constant())>::value;
        else
          return std::decay_t<decltype(std::declval<Trait>().get_constant_diagonal())>::value;
      }();

    static constexpr Likelihood status = constant_matrix<T> ? Likelihood::definitely : Likelihood::maybe;

    constexpr operator value_type() const noexcept { return value; }
    constexpr value_type operator()() const noexcept { return value; }
  };


  /**
   * \overload
   * \brief In this case, the constant is determined at runtime.
   */
#ifdef __cpp_concepts
  template<constant_matrix<Likelihood::maybe, CompileTimeStatus::unknown> T>
  struct constant_coefficient<T>
#else
  template<typename T>
  struct constant_coefficient<T, std::enable_if_t<constant_matrix<T, Likelihood::maybe, CompileTimeStatus::unknown>>>
#endif
  {
    template<typename Arg>
    explicit constexpr constant_coefficient(Arg&& arg) : value {[](Arg&& arg){
        using Trait = interface::SingleConstant<std::decay_t<T>>;
#ifdef __cpp_concepts
        if constexpr (requires(Trait& trait) { {trait.get_constant()} -> scalar_constant<CompileTimeStatus::unknown>; })
#else
        if constexpr (detail::is_constant_matrix<T, CompileTimeStatus::unknown>::value)
#endif
          return get_scalar_constant_value(Trait{std::forward<Arg>(arg)}.get_constant());
        else
          return get_scalar_constant_value(Trait{std::forward<Arg>(arg)}.get_constant_diagonal());
      }(std::forward<Arg>(arg))} {};

    using value_type = scalar_type_of_t<T>;
    using type = constant_coefficient;

    static constexpr Likelihood status = constant_matrix<T, Likelihood::definitely, CompileTimeStatus::unknown> ?
      Likelihood::definitely : Likelihood::maybe;

    constexpr operator value_type() const noexcept { return value; }
    constexpr value_type operator()() const noexcept { return value; }

  private:

    value_type value;
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
  template<constant_matrix<Likelihood::maybe> T>
#else
  template<typename T>
#endif
  constexpr auto constant_coefficient_v = constant_coefficient<T>::value;


  // --------------------------------------------------------- //
  //  constant_diagonal_coefficient, constant_diagonal_matrix  //
  // --------------------------------------------------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct is_diagonal_matrix : std::false_type {};

    template<typename T>
    struct is_diagonal_matrix<T, std::enable_if_t<interface::DiagonalTraits<std::decay_t<T>>::is_diagonal>>
      : std::true_type {};
  }
#endif


  /**
   * \brief Specifies that a type is a constant diagonal matrix, with the constant known at compile time.
   */
  template<typename T, Likelihood b = Likelihood::definitely, CompileTimeStatus c = CompileTimeStatus::known>
#ifdef __cpp_concepts
  concept constant_diagonal_matrix = indexible<T> and
    requires(interface::SingleConstant<std::decay_t<T>>& trait) { requires
      requires {
        {trait.get_constant_diagonal()} -> scalar_constant<c>;
        requires b != Likelihood::definitely or interface::DiagonalTraits<std::decay_t<T>>::is_diagonal or
          (not has_dynamic_dimensions<T> and detail::maybe_square_matrix_impl<T>(std::make_index_sequence<max_indices_of_v<T>>{})) or
          requires { requires std::decay_t<decltype(trait.get_constant_diagonal())>::status == Likelihood::definitely; };
      } or
      requires {
        {trait.get_constant()} -> scalar_constant<c>;
        requires (b != Likelihood::definitely or not has_dynamic_dimensions<T>) and
          (detail::maybe_one_by_one_matrix_impl<T>(std::make_index_sequence<max_indices_of_v<T>>{}) or
          (requires { requires are_within_tolerance(std::decay_t<decltype(trait.get_constant())>::value, 0); } and
            detail::maybe_square_matrix_impl<T>(std::make_index_sequence<max_indices_of_v<T>>{})));
      };
    };
#else
  constexpr bool constant_diagonal_matrix = indexible<T> and (
    (detail::is_constant_diagonal_matrix<T, c>::value and
      (b != Likelihood::definitely or detail::is_diagonal_matrix<T>::value or
        (not has_dynamic_dimensions<T> and detail::maybe_square_matrix_impl<T>(std::make_index_sequence<max_indices_of_v<T>>{})) or
        detail::constant_diagonal_status<T, Likelihood::definitely>::value)) or
    (detail::is_constant_matrix<T, c>::value and
      ((b != Likelihood::definitely or not has_dynamic_dimensions<T>) and
        (detail::maybe_one_by_one_matrix_impl<T>(std::make_index_sequence<max_indices_of_v<T>>{}) or
        (detail::is_specific_constant_matrix<T, 0>::value and
          detail::maybe_square_matrix_impl<T>(std::make_index_sequence<max_indices_of_v<T>>{}))))));
#endif


  /**
   * \brief The constant associated with T, assuming T is a \ref constant_diagonal_matrix.
   * \details Before using this value, always check if T is a \ref constant_diagonal_matrix, because the value be defined exist
   * even in cases where T is not actually a constant diagonal matrix.
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
   * \overload
   * \brief In this case, the constant can be derived at compile time.
   */
#ifdef __cpp_concepts
  template<constant_diagonal_matrix<Likelihood::maybe, CompileTimeStatus::known> T>
  struct constant_diagonal_coefficient<T>
#else
  template<typename T>
  struct constant_diagonal_coefficient<T, std::enable_if_t<constant_diagonal_matrix<T, Likelihood::maybe, CompileTimeStatus::known>>>
#endif
  {
    constexpr constant_diagonal_coefficient() = default;
    explicit constexpr constant_diagonal_coefficient(const std::decay_t<T>&) {};
    using value_type = scalar_type_of_t<T>;
    using type = constant_diagonal_coefficient;

    static constexpr value_type value =
      []{
        using Trait = interface::SingleConstant<std::decay_t<T>>;
#ifdef __cpp_concepts
        if constexpr (requires(Trait& trait) { {trait.get_constant_diagonal()} -> scalar_constant<CompileTimeStatus::known>; })
#else
        if constexpr (detail::is_constant_diagonal_matrix<T>::value)
#endif
          return std::decay_t<decltype(std::declval<Trait>().get_constant_diagonal())>::value;
        else
          return std::decay_t<decltype(std::declval<Trait>().get_constant())>::value;
      }();

    static constexpr Likelihood status = constant_diagonal_matrix<T> ? Likelihood::definitely : Likelihood::maybe;

    constexpr operator value_type() const noexcept { return value; }
    constexpr value_type operator()() const noexcept { return value; }
  };


  /**
   * \overload
   * \brief In this case, the constant can be derived at compile time.
   */
#ifdef __cpp_concepts
  template<constant_diagonal_matrix<Likelihood::maybe, CompileTimeStatus::unknown> T>
  struct constant_diagonal_coefficient<T>
#else
  template<typename T>
  struct constant_diagonal_coefficient<T, std::enable_if_t<constant_diagonal_matrix<T, Likelihood::maybe, CompileTimeStatus::unknown>>>
#endif
  {
    template<typename Arg>
    explicit constexpr constant_diagonal_coefficient(Arg&& arg) : value {[](Arg&& arg){
        using Trait = interface::SingleConstant<std::decay_t<T>>;
#ifdef __cpp_concepts
        if constexpr (requires(Trait& trait) { {trait.get_constant_diagonal()} -> scalar_constant<CompileTimeStatus::unknown>; })
#else
        if constexpr (detail::is_constant_diagonal_matrix<T, CompileTimeStatus::unknown>::value)
#endif
          return get_scalar_constant_value(Trait{std::forward<Arg>(arg)}.get_constant_diagonal());
        else
          return get_scalar_constant_value(Trait{std::forward<Arg>(arg)}.get_constant());
      }(std::forward<Arg>(arg))} {};

    using value_type = scalar_type_of_t<T>;
    using type = constant_diagonal_coefficient;

    static constexpr Likelihood status = constant_diagonal_matrix<T, Likelihood::definitely, CompileTimeStatus::unknown> ?
      Likelihood::definitely : Likelihood::maybe;

    constexpr operator value_type() const noexcept { return value; }
    constexpr value_type operator()() const noexcept { return value; }

  private:

    value_type value;
  };


  /**
   * \brief Deduction guide for \ref constant_diagonal_coefficient.
   */
  template<typename T>
  explicit constant_diagonal_coefficient(T&&) -> constant_diagonal_coefficient<std::decay_t<T>>;


  /// Helper template for constant_diagonal_coefficient.
#ifdef __cpp_concepts
  template<constant_diagonal_matrix<Likelihood::maybe> T>
#else
  template<typename T>
#endif
  constexpr auto constant_diagonal_coefficient_v = constant_diagonal_coefficient<T>::value;


  // ------------- //
  //  zero_matrix  //
  // ------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, Likelihood b, typename = void>
    struct is_zero_matrix : std::false_type {};

    template<typename T, Likelihood b>
    struct is_zero_matrix<T, b, std::enable_if_t<constant_matrix<T, b, CompileTimeStatus::known>>>
      : std::bool_constant<are_within_tolerance(constant_coefficient_v<T>, 0)> {};
  }
#endif


  /**
   * \brief Specifies that a type is known at compile time to be a constant matrix of value zero.
   */
  template<typename T, Likelihood b = Likelihood::definitely>
#ifdef __cpp_concepts
  concept zero_matrix = constant_matrix<T, b, CompileTimeStatus::known> and are_within_tolerance(constant_coefficient_v<T>, 0);
#else
  constexpr bool zero_matrix = detail::is_zero_matrix<T, b>::value;
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
    struct is_identity_matrix<T, b, std::enable_if_t<constant_diagonal_matrix<T, b>>>
      : std::bool_constant<are_within_tolerance(constant_diagonal_coefficient_v<T>, 1)> {};
  }
#endif

  /**
   * \brief Specifies that a type is an identity matrix.
   */
  template<typename T, Likelihood b = Likelihood::definitely>
#ifdef __cpp_concepts
  concept identity_matrix = constant_diagonal_matrix<T, b> and are_within_tolerance(constant_diagonal_coefficient_v<T>, 1);
#else
  constexpr bool identity_matrix = detail::is_identity_matrix<T, b>::value;
#endif


  // ----------------- //
  //  diagonal_matrix  //
  // ----------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, TriangleType t, typename = void>
    struct is_triangular_matrix : std::false_type {};

    template<typename T, TriangleType t>
    struct is_triangular_matrix<T, t, std::enable_if_t<interface::TriangularTraits<std::decay_t<T>>::triangle_type == t>>
      : std::true_type {};

    template<typename T, typename = void>
    struct is_hermitian_matrix : std::false_type {};

    template<typename T>
    struct is_hermitian_matrix<T, std::enable_if_t<interface::HermitianTraits<std::decay_t<T>>::is_hermitian>>
      : std::true_type {};

    template<typename T, typename = void>
    struct hermitian_adapter_triangle_type : std::integral_constant<TriangleType, TriangleType::none> {};

    template<typename T>
    struct hermitian_adapter_triangle_type<T, std::void_t<decltype(interface::HermitianTraits<std::decay_t<T>>::adapter_type)>>
      : std::integral_constant<TriangleType, interface::HermitianTraits<std::decay_t<T>>::adapter_type> {};
  } // namespace detail
#endif


  /**
   * \brief Specifies that a type is a diagonal matrix.
   */
  template<typename T, Likelihood b = Likelihood::definitely>
#ifdef __cpp_concepts
  concept diagonal_matrix = indexible<T> and (
    interface::DiagonalTraits<std::decay_t<T>>::is_diagonal or
    (interface::TriangularTraits<std::decay_t<T>>::triangle_type == TriangleType::diagonal) or
    (interface::HermitianTraits<std::decay_t<T>>::is_hermitian and
      interface::HermitianTraits<std::decay_t<T>>::adapter_type == TriangleType::diagonal) or
    constant_diagonal_matrix<T, b, CompileTimeStatus::any> or
    ((b != Likelihood::definitely or not has_dynamic_dimensions<T>) and
      detail::maybe_one_by_one_matrix_impl<T>(std::make_index_sequence<max_indices_of_v<T>>{})));
#else
  constexpr bool diagonal_matrix = indexible<T> and
    (detail::is_diagonal_matrix<T>::value or
    detail::is_triangular_matrix<T, TriangleType::diagonal>::value or
    (detail::is_hermitian_matrix<T>::value and detail::hermitian_adapter_triangle_type<T>::value == TriangleType::diagonal) or
    constant_diagonal_matrix<T, b, CompileTimeStatus::any> or
    ((b != Likelihood::definitely or not has_dynamic_dimensions<T>) and
      detail::maybe_one_by_one_matrix_impl<T>(std::make_index_sequence<max_indices_of_v<T>>{})));
#endif


  // ------------------ //
  //  diagonal_adapter  //
  // ------------------ //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, Likelihood b, typename = void>
    struct nested_matrix_is_order_1 : std::false_type {};

    template<typename T, Likelihood b>
    struct nested_matrix_is_order_1<T, b, std::enable_if_t<has_nested_matrix<T>>> : std::bool_constant<
      (max_tensor_order_of_v<nested_matrix_of_t<T>> >= 1) and
      (b != Likelihood::definitely or max_tensor_order_of_v<nested_matrix_of_t<T>> == 1)> {};
  }
#endif


  /**
   * \brief Specifies that a type is a diagonal adapter.
   * \tparam T A matrix or tensor.
   */
  template<typename T, Likelihood b = Likelihood::definitely>
#ifdef __cpp_concepts
  concept diagonal_adapter = diagonal_matrix<T, b> and has_nested_matrix<T> and
    (max_tensor_order_of_v<nested_matrix_of_t<T>> >= 1) and
    (b != Likelihood::definitely or max_tensor_order_of_v<nested_matrix_of_t<T>> == 1);
#else
  constexpr bool diagonal_adapter = diagonal_matrix<T> and has_nested_matrix<T> and
    detail::nested_matrix_is_order_1<T, b>::value;
#endif


  // ---------------------------- //
  //  diagonal_hermitian_adapter  //
  // ---------------------------- //

  /**
   * \brief Specifies that T is a hermitian matrix adapter that stores data along the diagonal.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept diagonal_hermitian_adapter = has_nested_matrix<T> and interface::HermitianTraits<std::decay_t<T>>::is_hermitian and
    (interface::HermitianTraits<std::decay_t<T>>::adapter_type == TriangleType::diagonal);
#else
  constexpr bool diagonal_hermitian_adapter = has_nested_matrix<T> and detail::is_hermitian_matrix<T>::value and
    (detail::hermitian_adapter_triangle_type<T>::value == TriangleType::diagonal);
#endif


  // ------------------------- //
  //  lower_hermitian_adapter  //
  // ------------------------- //

  /**
   * \brief Specifies that T is a hermitian matrix adapter that stores data in the lower-left triangle.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept lower_hermitian_adapter = has_nested_matrix<T> and interface::HermitianTraits<std::decay_t<T>>::is_hermitian and
    (interface::HermitianTraits<std::decay_t<T>>::adapter_type == TriangleType::lower);
#else
  constexpr bool lower_hermitian_adapter = has_nested_matrix<T> and detail::is_hermitian_matrix<T>::value and
    (detail::hermitian_adapter_triangle_type<T>::value == TriangleType::lower);
#endif


  // ------------------------- //
  //  upper_hermitian_adapter  //
  // ------------------------- //

  /**
   * \brief Specifies that T is a hermitian matrix adapter that stores data in the upper-right triangle.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept upper_hermitian_adapter = has_nested_matrix<T> and interface::HermitianTraits<std::decay_t<T>>::is_hermitian and
    (interface::HermitianTraits<std::decay_t<T>>::adapter_type == TriangleType::upper);
#else
  constexpr bool upper_hermitian_adapter = has_nested_matrix<T> and detail::is_hermitian_matrix<T>::value and
    (detail::hermitian_adapter_triangle_type<T>::value == TriangleType::upper);
#endif


  // ------------------- //
  //  hermitian_adapter  //
  // ------------------- //

  /**
   * \brief Specifies that a type is a hermitian matrix adapter.
   * \tparam T A matrix or tensor.
   */
  template<typename T>
#ifdef __cpp_concepts
  concept hermitian_adapter =
#else
  constexpr bool hermitian_adapter =
#endif
    diagonal_hermitian_adapter<T> or lower_hermitian_adapter<T> or upper_hermitian_adapter<T>;


  // --------------------------- //
  //  hermitian_adapter_type_of  //
  // --------------------------- //

  /**
   * \brief The TriangleType associated with the storage triangle of one or more matrices.
   */
  template<typename T, typename...Ts>
  struct hermitian_adapter_type_of
    : std::integral_constant<TriangleType,
      (diagonal_hermitian_adapter<T> and ... and diagonal_hermitian_adapter<Ts>) ? TriangleType::diagonal :
      (lower_hermitian_adapter<T> and ... and lower_hermitian_adapter<Ts>) ? TriangleType::lower :
      (upper_hermitian_adapter<T> and ... and upper_hermitian_adapter<Ts>) ? TriangleType::upper :
      TriangleType::none> {};


  /**
   * \brief The TriangleType associated with the storage triangle of a \ref hermitian_matrix.
   */
  template<typename T, typename...Ts>
  constexpr auto hermitian_adapter_type_of_v = hermitian_adapter_type_of<T, Ts...>::value;


  // ------------------ //
  //  hermitian_matrix  //
  // ------------------ //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, typename = void>
    struct imag_part_is_zero : std::false_type {};

    template<typename T>
    struct imag_part_is_zero<T, std::enable_if_t<imaginary_part(constant_coefficient<T>::value) == 0>>
      : std::true_type {};

    template<typename T, typename = void>
    struct diag_imag_part_is_zero : std::bool_constant<imag_part_is_zero<T>::value> {};

    template<typename T>
    struct diag_imag_part_is_zero<T, std::enable_if_t<imaginary_part(constant_diagonal_coefficient<T>::value) == 0>>
      : std::true_type {};

    template<typename T, Likelihood b, typename = void>
    struct is_inferred_hermitian_matrix : std::false_type {};

    template<typename T, Likelihood b>
    struct is_inferred_hermitian_matrix<T, b, std::enable_if_t<
      (not complex_number<typename scalar_type_of<T>::type> or zero_matrix<T, b> or diag_imag_part_is_zero<T>::value) and
      ((constant_matrix<T, b, CompileTimeStatus::any> and (b != Likelihood::definitely or not has_dynamic_dimensions<T>) and
          detail::maybe_square_matrix_impl<T>(std::make_index_sequence<max_indices_of_v<T>>{})) or
        diagonal_matrix<T, b>)>>
      : std::true_type {};
  };
#endif


  /**
   * \brief Specifies that a type is a hermitian matrix.
   * \T A matrix or tensor.
   */
  template<typename T, Likelihood b = Likelihood::definitely>
#ifdef __cpp_concepts
  concept hermitian_matrix = interface::HermitianTraits<std::decay_t<T>>::is_hermitian or
    ((not complex_number<scalar_type_of_t<T>> or zero_matrix<T, b> or
        imaginary_part(constant_coefficient_v<T>) == 0 or imaginary_part(constant_diagonal_coefficient_v<T>) == 0) and
      ((constant_matrix<T, b, CompileTimeStatus::any> and (b != Likelihood::definitely or not has_dynamic_dimensions<T>) and
          detail::maybe_square_matrix_impl<T>(std::make_index_sequence<max_indices_of_v<T>>{})) or
        diagonal_matrix<T, b>));
#else
  constexpr bool hermitian_matrix = detail::is_hermitian_matrix<T>::value or detail::is_inferred_hermitian_matrix<T, b>::value;
#endif


  // ------------------------- //
  //  lower_triangular_matrix  //
  // ------------------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, Likelihood b, typename = void>
    struct is_lower_triangular_matrix : std::false_type {};

    template<typename T, Likelihood b>
    struct is_lower_triangular_matrix<T, b, std::enable_if_t<indexible<T>>>
      : std::bool_constant<(interface::TriangularTraits<std::decay_t<T>>::triangle_type == TriangleType::lower) or
      diagonal_matrix<T, b>> {};
  }
#endif


  /**
   * \brief Specifies that a type is a lower-triangular matrix.
   */
  template<typename T, Likelihood b = Likelihood::definitely>
#ifdef __cpp_concepts
  concept lower_triangular_matrix =
    (indexible<T> and interface::TriangularTraits<std::decay_t<T>>::triangle_type == TriangleType::lower) or
    diagonal_matrix<T, b>;
#else
  constexpr bool lower_triangular_matrix = detail::is_lower_triangular_matrix<T, b>::value;
#endif


  // ------------------------- //
  //  upper_triangular_matrix  //
  // ------------------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, Likelihood b, typename = void>
    struct is_upper_triangular_matrix : std::false_type {};

    template<typename T, Likelihood b>
    struct is_upper_triangular_matrix<T, b, std::enable_if_t<indexible<T>>>
      : std::bool_constant<(interface::TriangularTraits<std::decay_t<T>>::triangle_type == TriangleType::upper) or
      diagonal_matrix<T, b>> {};
  }
#endif


  /**
   * \brief Specifies that a type is an upper-triangular matrix.
   */
  template<typename T, Likelihood b = Likelihood::definitely>
#ifdef __cpp_concepts
  concept upper_triangular_matrix =
    (indexible<T> and interface::TriangularTraits<std::decay_t<T>>::triangle_type == TriangleType::upper) or
    diagonal_matrix<T, b>;
#else
  constexpr bool upper_triangular_matrix = detail::is_upper_triangular_matrix<T, b>::value;
#endif


  // ------------------- //
  //  triangular_matrix  //
  // ------------------- //

  /**
   * \brief Specifies that a type is a triangular matrix (upper, lower, or diagonal).
   * \tparam T A matrix or tensor.
   */
  template<typename T, Likelihood b = Likelihood::definitely>
#ifdef __cpp_concepts
  concept triangular_matrix =
#else
  constexpr bool triangular_matrix =
#endif
    lower_triangular_matrix<T, b> or upper_triangular_matrix<T, b>;


  // ------------------ //
  //  triangle_type_of  //
  // ------------------ //

  /**
   * \brief The TriangleType associated with a matrix, or the common TriangleType associated with a set of matrices.
   */
  template<typename T, typename...Ts>
  struct triangle_type_of
    : std::integral_constant<TriangleType,
      (diagonal_matrix<T> and ... and diagonal_matrix<Ts>) ? TriangleType::diagonal :
      ((lower_triangular_matrix<T> and ... and lower_triangular_matrix<Ts>) ? TriangleType::lower :
      ((upper_triangular_matrix<T> and ... and upper_triangular_matrix<Ts>) ? TriangleType::upper :
      TriangleType::none))> {};


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
    template<typename T, Likelihood b, typename = void>
    struct nested_matrix_is_not_triangular : std::false_type {};

    template<typename T, Likelihood b>
    struct nested_matrix_is_not_triangular<T, b, std::enable_if_t<not triangular_matrix<nested_matrix_of_t<T>, not b> or
      triangle_type_of_v<nested_matrix_of_t<T>> != triangle_type_of_v<T>>> : std::true_type {};
  }
#endif


  /**
   * \brief Specifies that a type is a triangular adapter of triangle type triangle_type.
   * \tparam T A matrix or tensor.
   */
  template<typename T, Likelihood b = Likelihood::definitely>
#ifdef __cpp_concepts
  concept triangular_adapter = triangular_matrix<T, b> and has_nested_matrix<T> and
    (not diagonal_adapter<T, not b>) and
    (not triangular_matrix<nested_matrix_of_t<T>, not b> or triangle_type_of_v<nested_matrix_of_t<T>> != triangle_type_of_v<T>);
#else
  constexpr bool triangular_adapter = triangular_matrix<T, b> and has_nested_matrix<T> and
    (not diagonal_adapter<T, not b>) and detail::nested_matrix_is_not_triangular<T, b>::value;
#endif


  // --------------- //
  //  square_matrix  //
  // --------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, Likelihood b, typename = void>
    struct is_square_matrix : std::false_type {};

    template<typename T, Likelihood b>
    struct is_square_matrix<T, b, std::enable_if_t<indexible<T>>>
      : std::bool_constant<indexible<T> and (max_indices_of_v<T> > 0) and
        (b != Likelihood::definitely or not has_dynamic_dimensions<T>) and
        (max_indices_of_v<T> != 1 or dynamic_dimension<T, 0> or equivalent_to<coefficient_types_of_t<T, 0>, Axis>) and
        (max_indices_of_v<T> < 2 or detail::maybe_square_matrix_impl<T>(std::make_index_sequence<max_indices_of_v<T>>{}))> {};
  }
#endif


  /**
   * \brief Specifies that a matrix is square (i.e., has equivalent index descriptors along each dimension).
   */
  template<typename T, Likelihood b = Likelihood::definitely>
#ifdef __cpp_concepts
  concept square_matrix = (indexible<T> and (max_indices_of_v<T> > 0) and
    (b != Likelihood::definitely or not has_dynamic_dimensions<T>) and
    (max_indices_of_v<T> != 1 or dynamic_dimension<T, 0> or equivalent_to<coefficient_types_of_t<T, 0>, Axis>) and
    (max_indices_of_v<T> < 2 or detail::maybe_square_matrix_impl<T>(std::make_index_sequence<max_indices_of_v<T>>{}))) or
#else
  constexpr bool square_matrix = detail::is_square_matrix<T, b>::value or
#endif
    hermitian_matrix<T> or triangular_matrix<T>;


  // ------------------- //
  //  one_by_one_matrix  //
  // ------------------- //

#ifndef __cpp_concepts
  namespace detail
  {
    template<typename T, Likelihood b, typename = void>
    struct is_one_by_one_matrix : std::false_type {};

    template<typename T, Likelihood b>
    struct is_one_by_one_matrix<T, b, std::enable_if_t<indexible<T>>>
      : std::bool_constant<indexible<T> and (max_indices_of_v<T> > 0) and
        (b != Likelihood::definitely or not has_dynamic_dimensions<T>) and
        (max_indices_of_v<T> != 1 or dynamic_dimension<T, 0> or equivalent_to<coefficient_types_of_t<T, 0>, Axis>) and
        (max_indices_of_v<T> < 2 or detail::maybe_one_by_one_matrix_impl<T>(std::make_index_sequence<max_indices_of_v<T>>{}))> {};
  }
#endif


  /**
   * \brief Specifies that a type is a one-by-one matrix (i.e., one row and one column).
   */
  template<typename T, Likelihood b = Likelihood::definitely>
#ifdef __cpp_concepts
  concept one_by_one_matrix = indexible<T> and (max_indices_of_v<T> > 0) and
    (b != Likelihood::definitely or not has_dynamic_dimensions<T>) and
    (max_indices_of_v<T> != 1 or dynamic_dimension<T, 0> or equivalent_to<coefficient_types_of_t<T, 0>, Axis>) and
    (max_indices_of_v<T> < 2 or detail::maybe_one_by_one_matrix_impl<T>(std::make_index_sequence<max_indices_of_v<T>>{}));
#else
  constexpr bool one_by_one_matrix = detail::is_one_by_one_matrix<T, b>::value;
#endif


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
