/* This file is part of OpenKalman, a header-only C++ library for
 * Kalman filters and other recursive filters.
 *
 * Copyright (c) 2022 Christopher Lee Ogden <ogden@gatech.edu>
 *
 * This Source Code Form is subject to the terms of the Mozilla Public
 * License, v. 2.0. If a copy of the MPL was not distributed with this
 * file, You can obtain one at https://mozilla.org/MPL/2.0/.
 */

/**
 * \file
 * \brief Overloaded general array functions.
 */

#ifndef OPENKALMAN_ARRAY_FUNCTIONS_HPP
#define OPENKALMAN_ARRAY_FUNCTIONS_HPP

namespace OpenKalman
{
  using namespace interface;

  // ----------------- //
  //  n_ary_operation  //
  // ----------------- //

  namespace detail
  {
    template<typename T>
    struct is_plus : std::false_type {};

    template<typename T>
    struct is_plus<std::plus<T>> : std::true_type {};

    template<typename T>
    struct is_multiplies : std::false_type {};

    template<typename T>
    struct is_multiplies<std::multiplies<T>> : std::true_type {};


#ifdef __cpp_concepts
    template<typename Op, typename...Scalar>
#else
    template<typename Op, typename = void, typename...Scalar>
#endif
    struct is_constexpr_n_ary_function_impl : std::false_type {};


#ifdef __cpp_concepts
    template<typename Op, typename...Scalar> requires requires(Scalar...x) { Op{}(x...); }
    struct is_constexpr_n_ary_function_impl<Op, Scalar...>
#else
    template<typename Op, typename...Scalar>
    struct is_constexpr_n_ary_function_impl<Op, std::void_t<decltype(Op{}(std::declval<Scalar>()...))>, Scalar...>
#endif
      : std::true_type {};


    template<typename Op, typename...Scalar>
#ifdef __cpp_concepts
    struct is_constexpr_n_ary_function : is_constexpr_n_ary_function_impl<Op, Scalar...> {};
#else
    struct is_constexpr_n_ary_function : is_constexpr_n_ary_function_impl<Op, void, Scalar...> {};
#endif


    template<typename Operation, typename...Args, std::size_t...I>
    constexpr bool is_invocable_with_indices(std::index_sequence<I...>)
    {
      return std::is_invocable_v<Operation&&, Args&&..., decltype(I)...>;
    }


  template<typename Operation, std::size_t...I, typename...Args>
  constexpr decltype(auto) n_ary_invoke_op(Operation&& operation, std::index_sequence<I...> seq, Args&&...args)
  {
    if constexpr (is_invocable_with_indices<Operation, Args...>(seq))
      return std::forward<Operation>(operation)(std::forward<Args>(args)..., static_cast<decltype(I)>(0)...);
    else
      return std::forward<Operation>(operation)(std::forward<Args>(args)...);
  }


#ifdef __cpp_concepts
  template<typename Operation, std::size_t indices, typename...Args>
#else
  template<typename Operation, std::size_t indices, typename = void, typename...Args>
#endif
  struct n_ary_operator_traits_impl {};


#ifdef __cpp_concepts
  template<typename Operation, std::size_t indices, typename...Args>
  requires (is_invocable_with_indices<Operation, Args...>(std::make_index_sequence<indices> {})) or
    std::is_invocable_v<Operation&&, Args&&...>
  struct n_ary_operator_traits_impl<Operation, indices, Args...>
#else
  template<typename Operation, std::size_t indices, typename...Args>
  struct n_ary_operator_traits_impl<Operation, indices, std::enable_if_t<
    is_invocable_with_indices<Operation, Args...>(std::make_index_sequence<indices> {}) or
    std::is_invocable_v<Operation&&, Args&&...>>, Args...>
#endif
  {
    using type = decltype(n_ary_invoke_op(
      std::declval<Operation&&>(), std::make_index_sequence<indices> {}, std::declval<Args&&>()...));
  };


  template<typename Operation, std::size_t indices, typename...Args>
  struct n_ary_operator_traits
#ifdef __cpp_concepts
    : n_ary_operator_traits_impl<Operation, indices, Args...> {};
#else
    : n_ary_operator_traits_impl<Operation, indices, void, Args...> {};
#endif


#ifndef __cpp_concepts
    template<typename Operation, std::size_t Indices, typename = void, typename...Args>
    struct n_ary_operator_impl : std::false_type {};

    template<typename Operation, std::size_t Indices, typename...Args>
    struct n_ary_operator_impl<Operation, Indices, std::enable_if_t<(indexible<Args> and ...) and
      (std::is_invocable<Operation&&, typename std::add_lvalue_reference<typename scalar_type_of<Args>::type>::type...>::value or
        is_invocable_with_indices<Operation, typename std::add_lvalue_reference<typename scalar_type_of<Args>::type>::type...>(
          std::make_index_sequence<Indices> {}))>, Args...>
    : std::true_type {};
#endif


    template<typename Operation, std::size_t Indices, typename...Args>
#ifdef __cpp_concepts
    concept n_ary_operator = (indexible<Args> and ...) and
      (std::is_invocable_v<Operation&&, std::add_lvalue_reference_t<scalar_type_of_t<Args>>...> or
        is_invocable_with_indices<Operation, std::add_lvalue_reference_t<scalar_type_of_t<Args>>...>(
          std::make_index_sequence<Indices> {}));
#else
    constexpr bool n_ary_operator = n_ary_operator_impl<Operation, Indices, void, Args...>::value;
#endif


#ifdef __cpp_concepts
    template<typename T, typename DTup, typename Op, typename...Args>
#else
    template<typename T, typename DTup, typename Op, typename = void, typename...Args>
#endif
    struct interface_defines_n_ary_operation : std::false_type {};


    template<typename T, typename DTup, typename Op, typename...Args>
#ifdef __cpp_concepts
    requires requires(const DTup& d_tup, Op op, Args...args) {
      interface::ArrayOperations<std::decay_t<T>>::n_ary_operation(d_tup, op, std::forward<Args>(args)...);
    }
    struct interface_defines_n_ary_operation<T, DTup, Op, Args...>
#else
    struct interface_defines_n_ary_operation<T, DTup, Op, std::void_t<
      decltype(interface::ArrayOperations<std::decay_t<T>>::n_ary_operation_with_indices(
        std::declval<const DTup&>(), std::declval<Op>(), std::declval<Args>()...))>, Args...>
#endif
      : std::true_type {};


#ifdef __cpp_concepts
    template<typename T, typename DTup, typename Op, typename...Args>
#else
    template<typename T, typename DTup, typename Op, typename = void, typename...Args>
#endif
    struct interface_defines_n_ary_operation_with_indices : std::false_type {};


    template<typename T, typename DTup, typename Op, typename...Args>
#ifdef __cpp_concepts
    requires requires(const DTup& d_tup, Op op, Args...args) {
      interface::ArrayOperations<std::decay_t<T>>::n_ary_operation_with_indices(d_tup, op, std::forward<Args>(args)...);
    }
    struct interface_defines_n_ary_operation_with_indices<T, DTup, Op, Args...>
#else
    struct interface_defines_n_ary_operation_with_indices<T, DTup, Op, std::void_t<
      decltype(interface::ArrayOperations<std::decay_t<T>>::n_ary_operation_with_indices(
        std::declval<const DTup&>(), std::declval<Op>(), std::declval<Args>()...))>, Args...>
#endif
      : std::true_type {};


    template<typename Arg, std::size_t...I, typename...J>
    inline decltype(auto) n_ary_operation_get_element_impl(Arg&& arg, std::index_sequence<I...>, J...j)
    {
      if constexpr (sizeof...(I) == sizeof...(J))
        return get_element(std::forward<Arg>(arg), (j < get_index_dimension_of<I>(arg) ? j : 0)...);
      else
        return get_element(std::forward<Arg>(arg), [](auto dim, const auto& j_tup){
          auto j = std::get<I>(j_tup);
          if (j < dim) return j;
          else return 0;
        }(get_index_dimension_of<I>(arg), std::tuple {j...})...);
    }


    template<typename Operation, typename ArgsTup, std::size_t...ArgI, typename...J>
    inline auto n_ary_operation_get_element(Operation&& operation, ArgsTup&& args_tup, std::index_sequence<ArgI...>, J...j)
    {
      if constexpr (std::is_invocable_v<Operation&&, scalar_type_of_t<std::tuple_element_t<ArgI, std::decay_t<ArgsTup>>>..., J...>)
        return std::forward<Operation>(operation)(n_ary_operation_get_element_impl(
          std::get<ArgI>(std::forward<ArgsTup>(args_tup)),
          std::make_index_sequence<max_indices_of_v<std::tuple_element_t<ArgI, std::decay_t<ArgsTup>>>> {},
          j...)..., j...);
      else
        return std::forward<Operation>(operation)(n_ary_operation_get_element_impl(
          std::get<ArgI>(std::forward<ArgsTup>(args_tup)),
          std::make_index_sequence<max_indices_of_v<std::tuple_element_t<ArgI, std::decay_t<ArgsTup>>>> {},
          j...)...);
    }


    template<typename M, typename Operation, typename ArgsTup, typename...J>
    inline void n_ary_operation_iterate(M& m, Operation&& operation, ArgsTup&& args_tup, std::index_sequence<>, J...j)
    {
      set_element(m, n_ary_operation_get_element(std::forward<Operation>(operation), args_tup,
        std::make_index_sequence<std::tuple_size_v<ArgsTup>> {}, j...), j...);
    }


    template<typename M, typename Operation, typename ArgsTup, std::size_t I, std::size_t...Is, typename...J>
    inline void n_ary_operation_iterate(M& m, Operation&& operation, ArgsTup&& args_tup, std::index_sequence<I, Is...>, J...j)
    {
      for (std::size_t i = 0; i < get_index_dimension_of<I>(m); i++)
        n_ary_operation_iterate(m, operation, std::forward<ArgsTup>(args_tup), std::index_sequence<Is...> {}, j..., i);
    }


    template<typename PatternMatrix, typename...Ds, typename Op, typename...Args>
    static constexpr decltype(auto)
    n_ary_operation_with_broadcasting_impl(const std::tuple<Ds...>& tup, Op&& op, Args&&...args)
    {
      // zero_matrix:
      if constexpr (sizeof...(Args) > 0 and (zero_matrix<Args> and ...) and
        (is_plus<Op>::value or is_multiplies<Op>::value))
      {
        using Scalar = decltype(op(std::forward<Args>(args)...));
        return std::apply(
          [](auto&&...ds){ return make_zero_matrix_like<PatternMatrix, Scalar>(std::forward<decltype(ds)>(ds)...); },
          tup);
      }

      // constant_matrix:
      else if constexpr (sizeof...(Args) > 0 and (constant_matrix<Args> and ...) and
        is_constexpr_n_ary_function<Op, scalar_type_of_t<Args>...>::value)
      {

        constexpr auto c = Op{}(constant_coefficient_v<Args>...);
        using Scalar = std::decay_t<decltype(c)>;
# if __cpp_nontype_template_args >= 201911L
        return std::apply(
          [](auto&&...ds){ return make_constant_matrix_like<PatternMatrix, c, Scalar>(std::forward<decltype(ds)>(ds)...); },
          tup);
# else
        constexpr auto c_integral = static_cast<std::intmax_t>(c);
        if constexpr (are_within_tolerance(c, static_cast<Scalar>(c_integral)))
          return std::apply(
            [](auto&&...ds){
              return make_constant_matrix_like<PatternMatrix, c_integral, Scalar>(std::forward<decltype(ds)>(ds)...);
            },
            tup);
        else
          return make_self_contained(c * to_native_matrix<PatternMatrix>(std::apply(
            [](auto&&...ds){
              return make_constant_matrix_like<PatternMatrix, 1, Scalar>(std::forward<decltype(ds)>(ds)...);
            },
            tup)));
# endif
      }

      // other cases:
      else
      {
        constexpr std::make_index_sequence<max_indices_of_v<PatternMatrix>> seq;
        if constexpr (is_invocable_with_indices<Op&&, std::add_lvalue_reference_t<scalar_type_of_t<Args>>...>(seq) and
          detail::interface_defines_n_ary_operation_with_indices<PatternMatrix, std::tuple<Ds...>, Op&&, Args&&...>::value)
        {
          using Trait = interface::ArrayOperations<std::decay_t<PatternMatrix>>;
          return Trait::n_ary_operation_with_indices(tup, std::forward<Op>(op), std::forward<Args>(args)...);
        }
        else if constexpr (is_invocable_with_indices<Op&&, std::add_lvalue_reference_t<scalar_type_of_t<Args>>...>(seq) and
          detail::interface_defines_n_ary_operation<PatternMatrix, std::tuple<Ds...>, Op&&, Args&&...>::value)
        {
          using Trait = interface::ArrayOperations<std::decay_t<PatternMatrix>>;
          return Trait::n_ary_operation(tup, std::forward<Op>(op), std::forward<Args>(args)...);
        }
        else
        {
          using Scalar = std::decay_t<typename n_ary_operator_traits<Op, max_indices_of_v<PatternMatrix>,
            std::add_lvalue_reference_t<scalar_type_of_t<Args>>...>::type>;
          auto m = std::apply(
            [](auto&&...ds){
              return make_default_dense_writable_matrix_like<PatternMatrix, Scalar>(std::forward<decltype(ds)>(ds)...);
            },
            tup);
          n_ary_operation_iterate(m, std::forward<Op>(op), std::forward_as_tuple(std::forward<Args>(args)...), seq);
          return m;
        }
      }
    }


    template<typename...Ds, typename Op, typename Arg, typename...Args>
    static constexpr auto
    n_ary_operation_with_broadcasting_impl(const std::tuple<Ds...>& tup, Op&& op, Arg&& arg, Args&&...args)
    {
      return n_ary_operation_with_broadcasting_impl<Arg>(tup, std::forward<Op>(op), std::forward<Arg>(arg), std::forward<Args>(args)...);
    }


#ifdef __cpp_concepts
    template<typename DTup, typename Arg, std::size_t...indices>
#else
    template<typename DTup, typename Arg, typename = void, std::size_t...indices>
#endif
    struct n_ary_argument_index : std::false_type {};


#ifdef __cpp_concepts
    template<typename DTup, typename Arg, std::size_t...indices> requires (max_indices_of_v<Arg> <= sizeof...(indices)) and
      ((dimension_size_of_v<std::tuple_element_t<indices, DTup>> == dynamic_size or
      index_dimension_of_v<Arg, indices> == dynamic_size or
      equivalent_to<coefficient_types_of_t<Arg, indices>, std::tuple_element_t<indices, DTup>> or
      equivalent_to_uniform_dimension_type_of<coefficient_types_of_t<Arg, indices>, std::tuple_element_t<indices, DTup>> or
      (indices >= max_indices_of_v<Arg> and has_uniform_dimension_type<std::tuple_element_t<indices, DTup>>)) and ...)
    struct n_ary_argument_index<DTup, Arg, indices...>
#else
    template<typename DTup, typename Arg, std::size_t...indices>
    struct n_ary_argument_index<DTup, Arg, std::enable_if_t<(max_indices_of<Arg>::value <= sizeof...(indices)) and
      ((dimension_size_of<typename std::tuple_element<indices, DTup>::type>::value == dynamic_size or
      index_dimension_of<Arg, indices>::value == dynamic_size or
      equivalent_to<typename coefficient_types_of<Arg, indices>::type, typename std::tuple_element<indices, DTup>::type> or
      equivalent_to_uniform_dimension_type_of<typename coefficient_types_of<Arg, indices>::type,
        typename std::tuple_element<indices, DTup>::type> or
      (indices >= max_indices_of<Arg>::value and has_uniform_dimension_type<typename std::tuple_element<indices, DTup>::type>)) and ...)>, indices...>
#endif
    : std::true_type {};


    template<typename DTup, typename Arg, std::size_t...indices>
    constexpr bool n_ary_argument_impl(std::index_sequence<indices...>)
    {
# ifdef __cpp_concepts
      return n_ary_argument_index<DTup, Arg, indices...>::value;
# else
      return n_ary_argument_index<DTup, Arg, void, indices...>::value;
# endif
    }


    // Arg is a valid argument to n_ary_operation
    template<typename Arg, typename...Ds>
#ifdef __cpp_concepts
    concept n_ary_argument =
#else
    constexpr bool n_ary_argument =
#endif
      indexible<Arg> and (n_ary_argument_impl<std::tuple<Ds...>, Arg>(std::make_index_sequence<sizeof...(Ds)> {}));


    // Check that runtime dimensions of argument Arg is compatible with index descriptors Ds.
    template<typename DTup, typename Arg, std::size_t...indices>
    inline void check_n_ary_rt_dims(const DTup& d_tup, const Arg& arg, std::index_sequence<indices...>)
    {
      (([](const DTup& d_tup, const Arg& arg){
        if constexpr (dynamic_dimension<Arg, indices> or dynamic_index_descriptor<std::tuple_element_t<indices, DTup>>)
        {
          auto arg_d = get_dimensions_of<indices>(arg);
          auto tup_d = std::get<indices>(d_tup);

          if (arg_d == tup_d) return;
          else if (get_dimension_size_of(arg_d) == 1 and
            replicate_index_descriptor<scalar_type_of_t<Arg>>(arg_d, get_dimension_size_of(tup_d)) == tup_d) return;
          else throw std::logic_error {"In an argument to n_ary_operation, the dimension of index " +
            std::to_string(indices) + " is " + std::to_string(get_dimension_size_of(arg_d)) + ", but should be 1 " +
            (get_dimension_size_of(tup_d) == 1 ? "" : "or " + std::to_string(get_dimension_size_of(tup_d))) +
            "(the dimension of index " + std::to_string(indices) + " of the PatternMatrix template argument)"};
        }
      })(d_tup, arg),...);
    }

  } // namespace detail


  /**
   * \brief Perform a component-wise n-ary operation, using broadcasting to match the size of a pattern matrix.
   * \details This overload is for unary, binary, and higher n-ary operations. Examples:
   * - Unary operation, no broadcasting:
   *   \code
   *     auto ds32 = std::tuple {Dimensions<3>{}, Dimensions<2>{}};
   *     auto op1 = [](auto arg){return 3 * arg;};
   *     auto M = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
   *     auto m32 = make_dense_writable_matrix_from<M>(ds32, 1, 2, 3, 4, 5, 6);
   *     std::cout << n_ary_operation(ds32, op1, m32) << std::endl;
   *   \endcode
   *   Output:
   *   \code
   *     3, 6,
   *     9, 12,
   *     15, 18
   *   \endcode
   * - Unary operation, broadcasting:
   *   \code
   *     auto ds31 = std::tuple {Dimensions<3>{}, Dimensions<1>{}};
   *     auto m31 = make_dense_writable_matrix_from<M>(ds31, 1, 2, 3);
   *     std::cout << n_ary_operation(ds32, op1, m31) << std::endl;
   *   \endcode
   *   Output:
   *   \code
   *     3, 3,
   *     6, 6,
   *     9, 9
   *   \endcode
   * - Binary operation, no broadcasting:
   *   \code
   *     auto op2 = [](auto arg1, auto arg2){return 3 * arg1 + arg2;};
   *     std::cout << n_ary_operation(ds32, op2, m32, 2 * m32) << std::endl;
   *   \endcode
   *   Output:
   *   \code
   *     5, 10,
   *     15, 20,
   *     25, 30
   *   \endcode
   * - Binary operation, broadcasting:
   *   \code
   *     std::cout << n_ary_operation(ds32, op2, m31, 2 * m32) << std::endl;
   *   \endcode
   *   Output:
   *   \code
   *     5, 7,
   *     12, 14,
   *     19, 21
   *   \endcode
   * - Binary operation, broadcasting, with indices:
   *   \code
   *     auto op2b = [](auto arg1, auto arg2, std::size_t row, std::size_t col){return 3 * arg1 + arg2 + row + col;};
   *     std::cout << n_ary_operation(ds32, op2b, m31, 2 * m32) << std::endl;
   *   \endcode
   *   Output:
   *   \code
   *     5, 8,
   *     13, 16,
   *     21, 24
   *   \endcode
   * \tparam Ds Index descriptors defining the size of the result
   * \tparam Operation The n-ary operation taking n arguments and, optionally, a set of indices indicating the location
   * within the result. The number of indices, if any, must match the number of indices in the result.
   * \tparam Args The arguments
   * \return A matrix or array in which each component is the result of calling Operation on corresponding components
   * from each of the arguments, in the order specified.
   */
#ifdef __cpp_concepts
  template<index_descriptor...Ds, typename Operation, detail::n_ary_argument<Ds...>...Args>
  requires (sizeof...(Args) > 0) and detail::n_ary_operator<Operation, sizeof...(Ds), Args...>
#else
  template<typename...Ds, typename Operation, typename...Args, std::enable_if_t<(sizeof...(Args) > 0) and
    (index_descriptor<Ds> and ...) and detail::n_ary_operator<Operation, sizeof...(Ds), Args...> and
    (detail::n_ary_argument<Args, Ds...> and ...), int> = 0>
#endif
  constexpr decltype(auto)
  n_ary_operation(const std::tuple<Ds...>& d_tup, Operation&& operation, Args&&...args)
  {
    if constexpr (((dimension_size_of_v<Ds> == dynamic_size) or ...) or (has_dynamic_dimensions<Args> or ...))
      ((has_dynamic_dimensions<Args> ?
        detail::check_n_ary_rt_dims(d_tup, args, std::make_index_sequence<sizeof...(Ds)> {}) : void(0)), ...);

    return detail::n_ary_operation_with_broadcasting_impl(d_tup, std::forward<Operation>(operation), std::forward<Args>(args)...);
  }


  namespace detail
  {
    template<std::size_t I, typename Arg, typename...Args>
    constexpr auto find_max_runtime_dims_impl(const Arg& arg, const Args&...args)
    {
      if constexpr (sizeof...(Args) == 0)
      {
        return get_index_dimension_of<I>(arg);
      }
      else
      {
        auto dim0 = get_dimension_size_of(get_dimensions_of<I>(arg));
        auto dim = get_dimension_size_of(find_max_runtime_dims_impl<I>(args...));

        if (dim0 == dim or dim == 1) return dim0;
        else if (dim0 == 1) return dim;
        else throw std::logic_error {"In an argument to n_ary_operation, the dimension of index " +
          std::to_string(I) + " is " + std::to_string(dim0) + ", which is not 1 and does not match index " +
          std::to_string(I) + " of a later argument, which is " + std::to_string(dim)};
      }
    }


    template<std::size_t I, typename...Args>
    constexpr auto find_max_dims_impl(const Args&...args)
    {
      constexpr auto max_stat_dim = std::max({(dynamic_dimension<Args, I> ? 0 : index_dimension_of_v<Args, I>)...});
      constexpr auto dim = max_stat_dim == 0 ? dynamic_size : max_stat_dim;

      if constexpr (((not dynamic_dimension<Args, I> and (index_dimension_of_v<Args, I> == 0 or
          (index_dimension_of_v<Args, I> != 1 and index_dimension_of_v<Args, I> != dim))) or ...))
        throw std::logic_error {"The dimension of arguments to n_ary_operation should be either "
          "1 or the maximum dimension among the arguments. Instead, the argument dimensions are" +
          ((" " + std::to_string(index_dimension_of_v<Args, I>) + " (index " + std::to_string(I) + ")") + ...)};

      if constexpr ((dim != dynamic_size and dim > 1) or (dim == 1 and not (dynamic_dimension<Args, I> or ...)))
        return Dimensions<dim>{};
      else
        return Dimensions<dynamic_size>{find_max_runtime_dims_impl<I>(args...)};
    }


    template<std::size_t...I, typename...Args>
    constexpr auto find_max_dims(std::index_sequence<I...>, const Args&...args)
    {
      return std::tuple {find_max_dims_impl<I>(args...)...};
    }


    // Args is a valid set of arguments to n_ary_operation
    template<typename...Args>
#ifdef __cpp_concepts
    concept n_ary_arguments =
#else
    constexpr bool n_ary_arguments =
#endif
      (indexible<Args> and ...) and
      (n_ary_argument_impl<decltype(detail::find_max_dims(
          std::make_index_sequence<std::max({max_indices_of_v<Args>...})> {}, std::declval<Args>()...)), Args>(
            std::make_index_sequence<std::max({max_indices_of_v<Args>...})> {}) and ...);


    template<typename Arg, std::size_t...I>
    constexpr decltype(auto) n_ary_get_element_0(Arg&& arg, std::index_sequence<I...>)
    {
      return get_element(std::forward<Arg>(arg), (I * 0)...);
    }


    //// operation_returns_lvalue_reference ////

#ifdef __cpp_concepts
    template<typename Operation, typename...Args>
#else
    template<typename Operation, typename = void, typename...Args>
#endif
    struct operation_returns_lvalue_reference_impl : std::false_type {};


#ifdef __cpp_concepts
    template<typename Operation, typename Arg>
    requires (not std::is_const_v<Arg>) and
      std::is_lvalue_reference_v<typename n_ary_operator_traits<Operation, max_indices_of_v<Arg>,
        std::add_lvalue_reference_t<scalar_type_of_t<Arg>>>::type>
    struct operation_returns_lvalue_reference_impl<Operation, Arg&>
#else
    template<typename Operation, typename Arg>
    struct operation_returns_lvalue_reference_impl<Operation, std::enable_if_t<
      (not std::is_const<Arg>::value) and
      std::is_lvalue_reference<typename n_ary_operator_traits<Operation, max_indices_of<Arg>::value,
        typename std::add_lvalue_reference<typename scalar_type_of<Arg>::type>::type>::type>::value>, Arg&>
#endif
    : std::true_type {};


    template<typename Operation, typename...Args>
#ifdef __cpp_concepts
    concept operation_returns_lvalue_reference = operation_returns_lvalue_reference_impl<Operation, Args...>::value;
#else
    constexpr bool operation_returns_lvalue_reference = operation_returns_lvalue_reference_impl<Operation, void, Args...>::value;
#endif


  //// operation_returns_void ////

#ifdef __cpp_concepts
    template<typename Operation, typename...Args>
#else
    template<typename Operation, typename = void, typename...Args>
#endif
    struct operation_returns_void_impl : std::false_type {};


#ifdef __cpp_concepts
    template<typename Operation, typename Arg>
    requires (not std::is_const_v<Arg>) and
      std::is_void_v<typename n_ary_operator_traits<Operation, max_indices_of_v<Arg>,
        std::add_lvalue_reference_t<scalar_type_of_t<Arg>>>::type>
    struct operation_returns_void_impl<Operation, Arg&>
#else
    template<typename Operation, typename Arg>
    struct operation_returns_void_impl<Operation, std::enable_if_t<
      (not std::is_const_v<Arg>) and
      std::is_void_v<typename n_ary_operator_traits<Operation, max_indices_of_v<Arg>,
        std::add_lvalue_reference_t<scalar_type_of_t<Arg>>>::type>>, Arg&>
#endif
    : std::true_type {};


    template<typename Operation, typename...Args>
#ifdef __cpp_concepts
    concept operation_returns_void = operation_returns_void_impl<Operation, Args...>::value;
#else
    constexpr bool operation_returns_void = operation_returns_void_impl<Operation, void, Args...>::value;
#endif


    template<typename Operation, typename Arg, typename...J>
    inline void unary_operation_in_place_impl(Operation&& operation, Arg& arg, std::index_sequence<>, J...j)
    {
      if constexpr (std::is_invocable_v<Operation&&, std::add_lvalue_reference_t<scalar_type_of_t<Arg>>, J...>)
        std::forward<Operation>(operation)(get_element(arg, j...), j...);
      else
        std::forward<Operation>(operation)(get_element(arg, j...));
    }


    template<typename Operation, typename Arg, std::size_t I, std::size_t...Is, typename...J>
    inline void unary_operation_in_place_impl(Operation&& operation, Arg& arg, std::index_sequence<I, Is...>, J...j)
    {
      for (std::size_t i = 0; i < get_index_dimension_of<I>(arg); i++)
      {
        unary_operation_in_place_impl(operation, arg, std::index_sequence<Is...> {}, j..., i);
      }
    }

  } // namespace detail


  /**
   * \overload
   * \brief Perform a component-wise n-ary operation, using broadcasting if necessary to make the arguments the same size.
   * \details Each of the arguments may be expanded by broadcasting. The result will derive each dimension from the
   * largest corresponding dimension among the arguments.
   * There are additional input options for unary operations: the operation may return either a scalar value, an
   * lvalue reference, or void. Examples:
   * - Binary operation, broadcasting:
   *   \code
   *     auto M = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
   *     auto op2a = [](auto arg1, auto arg2){return 3 * arg1 + arg2;};
   *     auto m31 = make_dense_writable_matrix_from<M>(ds31, 1, 2, 3);
   *     auto m32 = make_dense_writable_matrix_from<M>(ds32, 1, 2, 3, 4, 5, 6);
   *     std::cout << n_ary_operation(op2a, m31, 2 * m32) << std::endl;
   *   \endcode
   *   Output:
   *   \code
   *     5, 7,
   *     12, 14,
   *     19, 21
   *   \endcode
   * - Binary operation, broadcasting, with indices:
   *   \code
   *     auto op2b = [](auto arg1, auto arg2, std::size_t row, std::size_t col){return 3 * arg1 + arg2 + row + col;};
   *     std::cout << n_ary_operation(op2b, m31, 2 * m32) << std::endl;
   *   \endcode
   *   Output:
   *   \code
   *     5, 8,
   *     13, 16,
   *     21, 24
   *   \endcode
   * - Unary operation, with indices:
   *   \code
   *     auto op1a = [](auto& arg, std::size_t row, std::size_t col){return arg + row + col;};
   *     std::cout << n_ary_operation(op1a, m32) << std::endl;
   *   \endcode
   *   Output:
   *   \code
   *     1, 3,
   *     4, 6,
   *     7, 9
   *   \endcode
   * - Unary operation, with indices, in-place operation returning lvalue reference:
   *   \code
   *     auto op1b = [](auto& arg, std::size_t row, std::size_t col){return arg += row + col;};
   *     std::cout << n_ary_operation(op1b, m32) << std::endl;
   *   \endcode
   *   Output:
   *   \code
   *     1, 3,
   *     4, 6,
   *     7, 9
   *   \endcode
   * - Unary operation, with indices, in-place operation returning void:
   *   \code
   *     auto op1c = [](auto& arg, std::size_t row, std::size_t col){arg += row + col;};
   *     std::cout << n_ary_operation(op1c, m32) << std::endl;
   *   \endcode
   *   Output:
   *   \code
   *     1, 3,
   *     4, 6,
   *     7, 9
   *   \endcode
   * \tparam Operation The n-ary operation taking n arguments and, optionally, a set of indices. The operation may
   * return one of the following:
   * - a scalar value;
   * - an lvalue reference to a scalar element within the argument; or
   * - void (for example, if the operation works on an lvalue reference)
   * \tparam Args The arguments
   * \return A matrix or array in which each component is the result of calling Operation on corresponding components
   * from each of the arguments, in the order specified.
   */
#ifdef __cpp_concepts
  template<typename Operation, indexible...Args> requires (sizeof...(Args) > 0) and detail::n_ary_arguments<Args...> and
    detail::n_ary_operator<Operation, std::max({max_indices_of_v<Args>...}), Args...>
#else
  template<typename Operation, typename...Args, std::enable_if_t<(indexible<Args> and ... and (sizeof...(Args) > 0)) and
    detail::n_ary_arguments<Args...> and detail::n_ary_operator<Operation, std::max({max_indices_of_v<Args>...}), Args...>, int> = 0>
#endif
  constexpr decltype(auto)
  n_ary_operation(Operation&& operation, Args&&...args)
  {
    if constexpr (detail::operation_returns_lvalue_reference<Operation&&, Args&&...>)
    {
      auto args_tup = std::forward_as_tuple(args...);
      auto& arg = std::get<0>(args_tup);
      using Arg = std::decay_t<decltype(arg)>;
      constexpr std::make_index_sequence<max_indices_of_v<Arg>> seq;

      using G = decltype(detail::n_ary_get_element_0(std::declval<Arg&>(), seq));
      static_assert(std::is_same_v<G, std::decay_t<G>&>, "Cannot use n_ary_operation with an operation that returns an "
        "lvalue reference unless get_element(...) returns a non-const lvalue reference.");

      n_ary_operation_iterate(arg, std::forward<Operation>(operation), std::move(args_tup), seq);
      return arg;
    }
    else if constexpr (detail::operation_returns_void<Operation&&, Args&&...>)
    {
      auto args_tup = std::forward_as_tuple(args...);
      auto& arg = std::get<0>(args_tup);
      using Arg = std::decay_t<decltype(arg)>;
      constexpr std::make_index_sequence<max_indices_of_v<Arg>> seq;

      using G = decltype(detail::n_ary_get_element_0(std::declval<Arg&>(), seq));
      static_assert(std::is_same_v<G, std::decay_t<G>&>, "Cannot use n_ary_operation with an operation that returns "
        "void unless get_element(...) returns a non-const lvalue reference.");

      detail::unary_operation_in_place_impl(std::forward<Operation>(operation), arg, seq);
      return arg;
    }
    else
    {
      constexpr auto max_indices = std::max({max_indices_of_v<Args>...});
      auto d_tup = detail::find_max_dims(std::make_index_sequence<max_indices> {}, args...);
      return detail::n_ary_operation_with_broadcasting_impl(std::move(d_tup), std::forward<Operation>(operation),
        std::forward<Args>(args)...);

    }
  }


  namespace detail
  {
    template<std::size_t I, std::size_t...indices>
    constexpr bool is_index_match()
    {
      return ((I == indices) or ...);
    }


    template<typename D_tup, std::size_t...indices, std::size_t...I>
    constexpr std::size_t count_index_dims(std::index_sequence<I...>)
    {
      return ([]{
        if constexpr (is_index_match<I, indices...>()) return dimension_size_of_v<std::tuple_element_t<I, D_tup>>;
        else return 1;
      }() * ... * 1);
    }


    template<typename T, std::size_t...indices, std::size_t...I>
    constexpr std::size_t pattern_index_dims(std::index_sequence<I...>)
    {
      return ([]{
        if constexpr (is_index_match<I, indices...>()) return index_dimension_of_v<T, I>;
        else return 1;
      }() * ... * 1);
    }


    template<typename PatternMatrix, typename Operation, typename Descriptors_tuple, typename Index_seq, typename K_seq, typename...Is>
    void nullary_set_elements(PatternMatrix& m, Operation&& op, const Descriptors_tuple&, Index_seq, K_seq, Is...is)
    {
      constexpr auto seq = std::make_index_sequence<max_indices_of_v<PatternMatrix>> {};
      if constexpr (detail::is_invocable_with_indices<Operation&&>(seq))
        set_element(m, op(is...), is...);
      else
        set_element(m, op(), is...);
    }


    template<std::size_t DsIndex, std::size_t...DsIndices, typename PatternMatrix, typename Operation,
      typename Descriptors_tuple, std::size_t...indices, std::size_t...Ks, typename...Is>
    void nullary_set_elements(PatternMatrix& m, Operation&& op, const Descriptors_tuple& ds_tup,
      std::index_sequence<indices...> index_seq, std::index_sequence<Ks...> k_seq, Is...is)
    {
      if constexpr (((DsIndex == indices) or ...))
      {
        constexpr std::size_t i = ((DsIndex == indices ? Ks : 0) + ...);
        nullary_set_elements<DsIndices...>(m, std::forward<Operation>(op), ds_tup, index_seq, k_seq, is..., i);
      }
      else
      {
        // Iterate through the dimensions of the current DsIndex and add set elements for each dimension iteratively.
        for (std::size_t i = 0; i < get_dimension_size_of(std::get<DsIndex>(ds_tup)); ++i)
        {
          nullary_set_elements<DsIndices...>(m, op, ds_tup, index_seq, k_seq, is..., i);
        }
      }
    }


    template<std::size_t CurrentOpIndex, std::size_t factor, typename PatternMatrix, typename Operations_tuple,
      typename Descriptors_tuple, typename UniqueIndicesSeq, std::size_t...AllDsIndices, typename K_seq>
    void nullary_iterate(PatternMatrix& m, Operations_tuple&& op_tup, const Descriptors_tuple& ds_tup,
      UniqueIndicesSeq unique_indices_seq, std::index_sequence<AllDsIndices...>, K_seq k_seq)
    {
      nullary_set_elements<AllDsIndices...>(m, std::get<CurrentOpIndex>(
        std::forward<Operations_tuple>(op_tup)), ds_tup, unique_indices_seq, k_seq);
    }


    template<std::size_t CurrentOpIndex, std::size_t factor, std::size_t index, std::size_t...indices,
      typename PatternMatrix, typename Operations_tuple, typename Descriptors_tuple, typename UniqueIndicesSeq, typename AllDsSeq,
      std::size_t...Ks, std::size_t...Js, typename...J_seqs>
    void nullary_iterate(PatternMatrix& m, Operations_tuple&& op_tup, const Descriptors_tuple& ds_tup,
      UniqueIndicesSeq unique_indices_seq, AllDsSeq all_ds_seq, std::index_sequence<Ks...>, std::index_sequence<Js...>,
      J_seqs...j_seqs)
    {
      constexpr std::size_t new_factor = factor / dimension_size_of_v<std::tuple_element_t<index, Descriptors_tuple>>;

      ((nullary_iterate<CurrentOpIndex + new_factor * Js, new_factor, indices...>(
        m, std::forward<Operations_tuple>(op_tup), ds_tup, unique_indices_seq, all_ds_seq,
        std::index_sequence<Ks..., Js>{}, j_seqs...)),...);
    }

  } // namespace detail


  /**
   * \overload
   * \brief Perform a component-wise nullary operation.
   * \details
   * - One operation for the entire matrix
   *   \code
   *     auto ds23 = std::tuple {Dimensions<2>{}, Dimensions<3>{}};
   *     auto M = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
   *     std::cout << n_ary_operation<M>(std::index_sequence<>{}, ds23, [](auto arg){return 7;}) << std::endl;
   *   \endcode
   *   Output:
   *   \code
   *     7, 7, 7,
   *     7, 7, 7
   *   \endcode
   * - One operation for each element
   *   \code
   *     std::cout << n_ary_operation<M>(std::index_sequence<0, 1>{}, ds23, []{return 4;}, []{return 5;}, []{return 6;}, []{return 7;}, []{return 8;}, []{return 9;});
   *   \endcode
   *   Output:
   *   \code
   *     4, 5, 6,
   *     7, 8, 9
   *   \endcode
   * - One operation for each row
   *   \code
   *     auto ds23a = std::tuple {Dimensions<2>{}, Dimensions{3}};
   *     std::cout << n_ary_operation<M>(std::index_sequence<0>{}, ds23a, []{return 5;}, []{return 6;});
   *   \endcode
   *   Output:
   *   \code
   *     5, 5, 5,
   *     6, 6, 6
   *   \endcode
   * - One operation for each column
   *   \code
   *     auto ds23b = std::tuple {Dimensions{2}, Dimensions<3>{}};
   *     std::cout << n_ary_operation<M>(std::index_sequence<1>{}, ds23b, []{return 5;}, []{return 6;}, []{return 7;});
   *   \endcode
   *   Output:
   *   \code
   *     5, 6, 7,
   *     5, 6, 7
   *   \endcode
   * \tparam PatternMatrix A matrix or array corresponding to the result type. Its dimensions need not match the specified dimensions Ds
   * \tparam indices The indices, if any, for which there is a distinct distribution for each slice based on that index.
   * \tparam Ds Index descriptors for each index the result
   * \tparam Operation The nullary operation taking n arguments
   * \return A matrix or array in which each component is the result of calling Operation with no arguments and which has
   * dimensions corresponding to Ds
   */
#ifdef __cpp_concepts
  template<indexible PatternMatrix, std::size_t...indices, index_descriptor...Ds,
    detail::n_ary_operator<max_indices_of_v<PatternMatrix>>...Operations>
  requires
    ((fixed_index_descriptor<std::tuple_element_t<indices, std::tuple<Ds...>>>) and ...) and
    (sizeof...(Operations) == detail::count_index_dims<std::tuple<Ds...>, indices...>(std::index_sequence_for<Ds...> {}))
#else
  template<typename PatternMatrix, std::size_t...indices, typename...Ds, typename...Operations, std::enable_if_t<
    indexible<PatternMatrix> and (index_descriptor<Ds> and ...) and
    ((fixed_index_descriptor<std::tuple_element_t<indices, std::tuple<Ds...>>>) and ...) and
    (sizeof...(Operations) == detail::count_index_dims<std::tuple<Ds...>, indices...>(std::index_sequence_for<Ds...> {})) and
    (detail::n_ary_operator<Operations, max_indices_of_v<PatternMatrix>> and ...), int> = 0>
#endif
  constexpr auto
  n_ary_operation(std::index_sequence<indices...>, const std::tuple<Ds...>& d_tup, Operations&&...operations)
  {
    constexpr std::make_index_sequence<max_indices_of_v<PatternMatrix>> seq;
    using Scalar = std::common_type_t<std::decay_t<
      typename detail::n_ary_operator_traits<Operations, max_indices_of_v<PatternMatrix>>::type>...>;

    // One operation for all elements combined:
    if constexpr (sizeof...(Operations) == 1)
    {
      return detail::n_ary_operation_with_broadcasting_impl<PatternMatrix>(d_tup, std::forward<Operations>(operations)...);
    }
    // One operation for each element (only if index descriptors are fixed):
    else if constexpr (((dimension_size_of_v<Ds> != dynamic_size) and ...) and
      sizeof...(operations) == (dimension_size_of_v<Ds> * ...) and
      not (detail::is_invocable_with_indices<Operations&&>(seq) or ...))
    {
      return make_dense_writable_matrix_from<PatternMatrix, Scalar>(d_tup, std::forward<Operations>(operations)()...);
    }
    else
    {
      auto m = std::apply([](const auto&...ds){ return make_default_dense_writable_matrix_like<PatternMatrix, Scalar>(ds...); }, d_tup);
      auto operations_tuple = std::forward_as_tuple(std::forward<Operations>(operations)...);
      detail::nullary_iterate<0, sizeof...(Operations), indices...>(
        m, std::move(operations_tuple), d_tup,
        std::index_sequence<indices...> {},
        std::index_sequence_for<Ds...> {},
        std::index_sequence<> {},
        std::make_index_sequence<dimension_size_of_v<std::tuple_element_t<indices, std::tuple<Ds...>>>> {}...);
      return m;
    }
  }


  /**
   * \overload
   * \brief Perform a component-wise nullary operation, using a single operation for all elements.
   * \details Example:
   * - One operation for the entire matrix
   *   \code
   *     auto ds23 = std::tuple {Dimensions<2>{}, Dimensions<3>{}};
   *     auto M = Eigen::Matrix<double, Eigen::Dynamic, Eigen::Dynamic>
   *     std::cout << n_ary_operation<M>(ds23, [](auto arg){return 7;}) << std::endl;
   *   \endcode
   *   Output:
   *   \code
   *     7, 7, 7,
   *     7, 7, 7
   *   \endcode
   * \tparam PatternMatrix A matrix or array corresponding to the result type. Its dimensions need not match the specified dimensions Ds
   * \tparam Ds Index descriptors for each index the result
   * \tparam Operation The nullary operation
   * \return A matrix or array in which each component is the result of calling Operation with no arguments and which has
   * dimensions corresponding to Ds
   */
#ifdef __cpp_concepts
  template<indexible PatternMatrix, index_descriptor...Ds, detail::n_ary_operator<max_indices_of_v<PatternMatrix>> Operation>
#else
  template<typename PatternMatrix, typename...Ds, typename Operation, std::enable_if_t<
    indexible<PatternMatrix> and (index_descriptor<Ds> and ...) and
    detail::n_ary_operator<Operation, max_indices_of_v<PatternMatrix>>, int> = 0>
#endif
  constexpr auto
  n_ary_operation(const std::tuple<Ds...>& d_tup, Operation&& operation)
  {
    return n_ary_operation<PatternMatrix>(std::index_sequence<>{}, d_tup, std::forward<Operation>(operation));
  }


  /**
   * \overload
   * \brief Perform a component-wise nullary operation, deriving the resulting size from a pattern matrix.
     * \details
     * - One operation for the entire matrix
     *   \code
     *     auto M = Eigen::Matrix<double, 2, 3>
     *     std::cout << n_ary_operation<M>(std::index_sequence<>{}, [](auto arg){return 7;}) << std::endl;
     *   \endcode
     *   Output:
     *   \code
     *     7, 7, 7,
     *     7, 7, 7
     *   \endcode
     * - One operation for each element
     *   \code
     *     std::cout << n_ary_operation<M>(std::index_sequence<0, 1>{}, []{return 4;}, []{return 5;}, []{return 6;}, []{return 7;}, []{return 8;}, []{return 9;});
     *   \endcode
     *   Output:
     *   \code
     *     4, 5, 6,
     *     7, 8, 9
     *   \endcode
     * - One operation for each row
     *   \code
     *     std::cout << n_ary_operation<M>(std::index_sequence<0>{}, []{return 5;}, []{return 6;});
     *   \endcode
     *   Output:
     *   \code
     *     5, 5, 5,
     *     6, 6, 6
     *   \endcode
     * - One operation for each column
     *   \code
     *     std::cout << n_ary_operation<M>(std::index_sequence<1>{}, []{return 5;}, []{return 6;}, []{return 7;});
     *   \endcode
     *   Output:
     *   \code
     *     5, 6, 7,
     *     5, 6, 7
     *   \endcode
   * \tparam PatternMatrix A matrix or array corresponding to the result type. Its dimensions need not match the specified dimensions Ds
   * \tparam indices The indices, if any, for which there is a distinct distribution for each slice based on that index.
   * \tparam Ds Index descriptors for each index the result
   * \tparam Operation The nullary operation
   * \return A matrix or array in which each component is the result of calling Operation with no arguments and which has
   * dimensions corresponding to Ds
   */
#ifdef __cpp_concepts
  template<indexible PatternMatrix, std::size_t...indices, detail::n_ary_operator<max_indices_of_v<PatternMatrix>>...Operations>
  requires (not has_dynamic_dimensions<PatternMatrix>) and
    (sizeof...(Operations) == detail::pattern_index_dims<PatternMatrix, indices...>(
      std::make_index_sequence<max_indices_of_v<PatternMatrix>> {}))
#else
  template<typename PatternMatrix, std::size_t...indices, typename...Operations, std::enable_if_t<
    indexible<PatternMatrix> and (not has_dynamic_dimensions<PatternMatrix>) and
    (sizeof...(Operations) == detail::pattern_index_dims<PatternMatrix, indices...>(
      std::make_index_sequence<max_indices_of_v<PatternMatrix>> {})) and
    (detail::n_ary_operator<Operations, max_indices_of_v<PatternMatrix>> and ...), int> = 0>
#endif
  constexpr auto
  n_ary_operation(std::index_sequence<indices...> seq, Operations&&...operations)
  {
    auto d_tup = get_all_dimensions_of<PatternMatrix>();
    return n_ary_operation<PatternMatrix>(seq, d_tup, std::forward<Operations>(operations)...);
  }


  /**
   * \overload
   * \brief Perform a component-wise nullary operation on all elements, deriving the resulting size from a pattern matrix.
     * \details
     * - One operation for the entire matrix
     *   \code
     *     auto M = Eigen::Matrix<double, 2, 3>
     *     std::cout << n_ary_operation<M>([](auto arg){return 7;}) << std::endl;
     *   \endcode
     *   Output:
     *   \code
     *     7, 7, 7,
     *     7, 7, 7
     *   \endcode
     * - One operation for each element
     *   \code
     *     std::cout << n_ary_operation<M>([]{return 4;}, []{return 5;}, []{return 6;}, []{return 7;}, []{return 8;}, []{return 9;});
     *   \endcode
     *   Output:
     *   \code
     *     4, 5, 6,
     *     7, 8, 9
     *   \endcode
     * - One operation for each row
     *   \code
     *     std::cout << n_ary_operation<M>([]{return 5;}, []{return 6;});
     *   \endcode
     *   Output:
     *   \code
     *     5, 5, 5,
     *     6, 6, 6
     *   \endcode
     * - One operation for each column
     *   \code
     *     std::cout << n_ary_operation<M>([]{return 5;}, []{return 6;}, []{return 7;});
     *   \endcode
     *   Output:
     *   \code
     *     5, 6, 7,
     *     5, 6, 7
     *   \endcode
   * \tparam PatternMatrix A matrix or array corresponding to the result type. Its dimensions need not match the specified dimensions Ds
   * \tparam Operation The nullary operation
   * \return A matrix or array in which each component is the result of calling Operation with no arguments and which has
   * dimensions corresponding to Ds
   */
#ifdef __cpp_concepts
  template<indexible PatternMatrix, detail::n_ary_operator<max_indices_of_v<PatternMatrix>> Operation>
  requires (not has_dynamic_dimensions<PatternMatrix>)
#else
  template<typename PatternMatrix, typename Operation, std::enable_if_t<
    indexible<PatternMatrix> and (not has_dynamic_dimensions<PatternMatrix>) and
    detail::n_ary_operator<Operation, max_indices_of_v<PatternMatrix>>, int> = 0>
#endif
  constexpr auto
  n_ary_operation(Operation&& operation)
  {
    return n_ary_operation<PatternMatrix>(std::index_sequence<>{}, std::forward<Operation>(operation));
  }


  // ----------- //
  //  randomize  //
  // ----------- //

  namespace detail
  {
#ifndef __cpp_concepts
    template<typename T, typename = void, typename = void>
    struct is_std_dist : std::false_type {};

    template<typename T>
    struct is_std_dist<T, std::void_t<typename T::result_type>, std::void_t<typename T::param_type>> : std::true_type {};
#endif


    template<typename random_number_generator>
    struct RandomizeGenerator
    {
      static auto& get()
      {
        static std::random_device rd;
        static std::decay_t<random_number_generator> gen {rd()};
        return gen;
      }
    };


    template<typename random_number_generator, typename distribution_type>
    struct RandomizeOp
    {
      template<typename G, typename D>
      RandomizeOp(G& g, D&& d) : generator{g}, distribution{std::forward<D>(d)} {}

      auto operator()() const
      {
        if constexpr (std::is_arithmetic_v<distribution_type>)
          return distribution;
        else
          return distribution(generator);
      }

    private:

      std::decay_t<random_number_generator>& generator;
      mutable std::decay_t<distribution_type> distribution;
    };


    template<typename G, typename D>
    RandomizeOp(G&, D&&) -> RandomizeOp<G, D>;

  } // namespace detail


  /**
   * \brief Create a matrix with random values selected from one or more random distributions.
   * \details This is essentially a specialized version of \ref n_ary_operation_with_indices with the unary operator
   * being a randomization function. The distributions are allocated to each element of the matrix, according to one
   * of the following options:
   *  - One distribution for all matrix elements. The following example constructs a 2-by-2 matrix (m) in which each
   *  element is a random value selected based on a distribution with mean 1.0 and standard deviation 0.3:
   *   \code
   *     using N = std::normal_distribution<double>;
   *     using Mat = Eigen::Matrix<double, 2, 2>;
   *     auto g = std::mt19937 {};
   *     Mat m = randomize<Mat>(g, std::index_sequence<>{}, std::tuple {2, 2}, N {1.0, 0.3}));
   *   \endcode
   *  - One distribution for each matrix element. The following code constructs a 2-by-2 matrix m containing
   *  random values around mean 1.0, 2.0, 3.0, and 4.0 (in row-major order), with standard deviations of
   *  0.3, 0.3, 0.0 (by default, since no s.d. is specified as a parameter), and 0.3:
   *   \code
   *     using N = std::normal_distribution<double>;
   *     auto m = randomize<Eigen::Matrix<double, 2, 2>>(g, std::index_sequence<0, 1>{},
   *       std::tuple {Dimensions<2>{}, Dimensions<2>{}}, N {1.0, 0.3}, N {2.0, 0.3}, 3.0, N {4.0, 0.3})));
   *   \endcode
   *  - One distribution for each row. The following code constructs a 3-by-2 (n) or 2-by-2 (o) matrices
   *  in which elements in each row are selected according to the three (n) or two (o) listed distribution
   *  parameters:
   *   \code
   *     auto n = randomize<Eigen::Matrix<double, 3, 2>>(g, std::index_sequence<0>{}, std::tuple {Dimensions<3>{}, 2},
   *       N {1.0, 0.3}, 2.0, N {3.0, 0.3})));
   *     auto o = randomize<Eigen::Matrix<double, 2, 2>>(g, std::index_sequence<0>{}, std::tuple {Dimensions<2>{}, 2},
   *       N {1.0, 0.3}, N {2.0, 0.3})));
   *   \endcode
   *   Note that in the case of o, there is an ambiguity as to whether the listed distributions correspond to rows
   *   or columns. In case of such an ambiguity, this function assumes that the parameters correspond to the rows.
   *  - One distribution for each column. The following code constructs 2-by-3 matrix p
   *  in which elements in each column are selected according to the three listed distribution parameters:
   *   \code
   *     auto p = randomize<Eigen::Matrix<double, 2, 3>>(g, std::index_sequence<1>{}, std::tuple {2, Dimensions<3>{}},
   *       N {1.0, 0.3}, 2.0, N {3.0, 0.3})));
   *   \endcode
   * \tparam PatternMatrix A matrix or array corresponding to the result type. Its dimensions need not match the
   * specified dimensions Ds
   * \tparam indices The indices, if any, for which there is a distinct distribution. If not provided, this can in some
   * cases be inferred from the number of Dists provided.
   * \tparam random_number_generator The random number generator (e.g., std::mt19937).
   * \tparam Ds Index descriptors for each index the result. They need not correspond to the dimensions of PatternMatrix.
   * \tparam Dists One or more distributions (e.g., std::normal_distribution<double>)
   * \sa n_ary_operation_with_indices
   */
#ifdef __cpp_concepts
  template<indexible PatternMatrix, std::uniform_random_bit_generator random_number_generator,
    std::size_t...indices, index_descriptor...Ds, typename...Dists>
  requires (sizeof...(Ds) == max_indices_of_v<PatternMatrix>) and
    ((fixed_index_descriptor<std::tuple_element_t<indices, std::tuple<Ds...>>>) and ...) and
    (sizeof...(Dists) == detail::count_index_dims<std::tuple<Ds...>, indices...>(std::index_sequence_for<Ds...> {})) and
    ((std::is_arithmetic_v<std::decay_t<Dists>> or
      requires { typename std::decay_t<Dists>::result_type; typename std::decay_t<Dists>::param_type; }) and ...)
#else
  template<typename PatternMatrix, typename random_number_generator, std::size_t...indices, typename...Ds,
    typename...Dists, std::enable_if_t<indexible<PatternMatrix> and (index_descriptor<Ds> and ...) and
    (sizeof...(Ds) == max_indices_of_v<PatternMatrix>) and
    ((fixed_index_descriptor<std::tuple_element_t<indices, std::tuple<Ds...>>>) and ...) and
    (sizeof...(Dists) == detail::count_index_dims<std::tuple<Ds...>, indices...>(std::index_sequence_for<Ds...> {})) and
    ((std::is_arithmetic_v<std::decay_t<Dists>> or detail::is_std_dist<std::decay_t<Dists>>::value) and ...), int> = 0>
#endif
  constexpr auto
  randomize(random_number_generator& gen, std::index_sequence<indices...> seq, const std::tuple<Ds...>& ds_tuple, Dists&&...dists)
  {
    auto ret = n_ary_operation<PatternMatrix>(seq, ds_tuple,
      detail::RandomizeOp {gen, (std::forward<Dists>(dists))}...);

    if constexpr (sizeof...(Dists) == 1)
      return make_dense_writable_matrix_from(std::move(ret));
    else
      return ret;
  }


  /**
   * \overload
   * \brief Create a matrix with random values, using a default random number engine.
   * \details The distributions are allocated to each element of the matrix, according to one of the following options:
   *  - One distribution for all matrix elements. The following example constructs a 2-by-2 matrix (m) in which each
   *  element is a random value selected based on a distribution with mean 1.0 and standard deviation 0.3:
   *   \code
   *     using N = std::normal_distribution<double>;
   *     using Mat = Eigen::Matrix<double, 2, 2>;
   *     Mat m = randomize<Mat>(std::index_sequence<>{}, std::tuple {2, 2}, N {1.0, 0.3}));
   *   \endcode
   *  - One distribution for each matrix element. The following code constructs a 2-by-2 matrix m containing
   *  random values around mean 1.0, 2.0, 3.0, and 4.0 (in row-major order), with standard deviations of
   *  0.3, 0.3, 0.0 (by default, since no s.d. is specified as a parameter), and 0.3:
   *   \code
   *     using N = std::normal_distribution<double>;
   *     auto m = randomize<Eigen::Matrix<double, 2, 2>>(std::index_sequence<0, 1>{},
   *       std::tuple {Dimensions<2>{}, Dimensions<2>{}}, N {1.0, 0.3}, N {2.0, 0.3}, 3.0, N {4.0, 0.3})));
   *   \endcode
   *  - One distribution for each row. The following code constructs a 3-by-2 (n) or 2-by-2 (o) matrices
   *  in which elements in each row are selected according to the three (n) or two (o) listed distribution
   *  parameters:
   *   \code
   *     auto n = randomize<Eigen::Matrix<double, 3, 2>>(std::index_sequence<0>{}, std::tuple {Dimensions<3>{}, 2},
   *       N {1.0, 0.3}, 2.0, N {3.0, 0.3})));
   *     auto o = randomize<Eigen::Matrix<double, 2, 2>>(std::index_sequence<0>{}, std::tuple {Dimensions<2>{}, 2},
   *       N {1.0, 0.3}, N {2.0, 0.3})));
   *   \endcode
   *   Note that in the case of o, there is an ambiguity as to whether the listed distributions correspond to rows
   *   or columns. In case of such an ambiguity, this function assumes that the parameters correspond to the rows.
   *  - One distribution for each column. The following code constructs 2-by-3 matrix p
   *  in which elements in each column are selected according to the three listed distribution parameters:
   *   \code
   *     auto p = randomize<Eigen::Matrix<double, 2, 3>>(std::index_sequence<1>{}, std::tuple {2, Dimensions<3>{}},
   *       N {1.0, 0.3}, 2.0, N {3.0, 0.3})));
   *   \endcode
   */
#ifdef __cpp_concepts
  template<indexible PatternMatrix, std::uniform_random_bit_generator random_number_generator = std::mt19937,
    std::size_t...indices, index_descriptor...Ds, typename...Dists>
  requires (sizeof...(Ds) == max_indices_of_v<PatternMatrix>) and
    ((fixed_index_descriptor<std::tuple_element_t<indices, std::tuple<Ds...>>>) and ...) and
    (sizeof...(Dists) == detail::count_index_dims<std::tuple<Ds...>, indices...>(std::index_sequence_for<Ds...> {})) and
    std::constructible_from<random_number_generator, typename std::random_device::result_type> and
    ((std::is_arithmetic_v<std::decay_t<Dists>> or
      requires { typename std::decay_t<Dists>::result_type; typename std::decay_t<Dists>::param_type; }) and ...)
#else
  template<typename PatternMatrix, typename random_number_generator = std::mt19937,
      std::size_t...indices, typename...Ds, typename...Dists,
    std::enable_if_t<indexible<PatternMatrix> and (index_descriptor<Ds> and ...) and
    (sizeof...(Ds) == max_indices_of_v<PatternMatrix>) and
    ((fixed_index_descriptor<std::tuple_element_t<indices, std::tuple<Ds...>>>) and ...) and
    (sizeof...(Dists) == detail::count_index_dims<std::tuple<Ds...>, indices...>(std::index_sequence_for<Ds...> {})) and
    std::is_constructible_v<random_number_generator, typename std::random_device::result_type> and
    ((std::is_arithmetic_v<std::decay_t<Dists>> or detail::is_std_dist<std::decay_t<Dists>>::value) and ...), int> = 0>
#endif
  constexpr auto
  randomize(std::index_sequence<indices...> seq, const std::tuple<Ds...>& d_tuple, Dists&&...dists)
  {
    auto& gen = detail::RandomizeGenerator<random_number_generator>::get();
    return randomize<PatternMatrix>(gen, seq, d_tuple, std::forward<Dists>(dists)...);
  }


  /**
   * \overload
   * \brief Create a matrix with random values, using a default random number engine.
   * \details The distributions are allocated to each element of the matrix, according to one of the following options:
   *  - One distribution for all matrix elements. The following example constructs a 2-by-2 matrix (m) in which each
   *  element is a random value selected based on a distribution with mean 1.0 and standard deviation 0.3:
   *   \code
   *     using N = std::normal_distribution<double>;
   *     using Mat = Eigen::Matrix<double, 2, 2>;
   *     Mat m = randomize<Mat>(std::tuple {2, 2}, N {1.0, 0.3}));
   *   \endcode
   *  - One distribution for each matrix element. The following code constructs a 2-by-2 matrix m containing
   *  random values around mean 1.0, 2.0, 3.0, and 4.0 (in row-major order), with standard deviations of
   *  0.3, 0.3, 0.0 (by default, since no s.d. is specified as a parameter), and 0.3:
   *   \code
   *     using N = std::normal_distribution<double>;
   *     auto m = randomize<Eigen::Matrix<double, 2, 2>>(std::tuple {Dimensions<2>{}, Dimensions<2>{}},
   *       N {1.0, 0.3}, N {2.0, 0.3}, 3.0, N {4.0, 0.3})));
   *   \endcode
   *  - One distribution for each row. The following code constructs a 3-by-2 (n) or 2-by-2 (o) matrices
   *  in which elements in each row are selected according to the three (n) or two (o) listed distribution
   *  parameters:
   *   \code
   *     auto n = randomize<Eigen::Matrix<double, 3, 2>>(std::tuple {Dimensions<3>{}, 2}, N {1.0, 0.3}, 2.0, N {3.0, 0.3})));
   *     auto o = randomize<Eigen::Matrix<double, 2, 2>>(std::tuple {Dimensions<2>{}, 2}, N {1.0, 0.3}, N {2.0, 0.3})));
   *   \endcode
   *   Note that in the case of o, there is an ambiguity as to whether the listed distributions correspond to rows
   *   or columns. In case of such an ambiguity, this function assumes that the parameters correspond to the rows.
   *  - One distribution for each column. The following code constructs 2-by-3 matrix p
   *  in which elements in each column are selected according to the three listed distribution parameters:
   *   \code
   *     auto p = randomize<Eigen::Matrix<double, 2, 3>>(std::tuple {2, Dimensions<3>{}}, N {1.0, 0.3}, 2.0, N {3.0, 0.3})));
   *   \endcode
   */
#ifdef __cpp_concepts
  template<indexible PatternMatrix, std::uniform_random_bit_generator random_number_generator = std::mt19937,
    index_descriptor...Ds, typename Dist>
  requires (sizeof...(Ds) == max_indices_of_v<PatternMatrix>) and
    std::constructible_from<random_number_generator, typename std::random_device::result_type> and
    (std::is_arithmetic_v<std::decay_t<Dist>> or
      requires { typename std::decay_t<Dist>::result_type; typename std::decay_t<Dist>::param_type; })
#else
  template<typename PatternMatrix, typename random_number_generator = std::mt19937, typename...Ds, typename Dist,
    std::enable_if_t<indexible<PatternMatrix> and (index_descriptor<Ds> and ...) and
    (sizeof...(Ds) == max_indices_of_v<PatternMatrix>) and
    std::is_constructible_v<random_number_generator, typename std::random_device::result_type> and
    (std::is_arithmetic_v<std::decay_t<Dist>> or detail::is_std_dist<std::decay_t<Dist>>::value), int> = 0>
#endif
  constexpr auto
  randomize(const std::tuple<Ds...>& d_tuple, Dist&& dist)
  {
    return randomize<PatternMatrix, random_number_generator>(std::index_sequence<>{}, d_tuple, std::forward<Dist>(dist));
  }


  /**
   * \overload
   * \brief Fill a fixed-sized matrix with random values selected from one or more random distributions.
   * \details The distributions are allocated to each element of the matrix, according to one of the following options:
   *  - One distribution for all matrix elements. The following example constructs a 2-by-2 matrix (m) in which each
   *  element is a random value selected based on a distribution with mean 1.0 and standard deviation 0.3:
   *   \code
   *     using N = std::normal_distribution<double>;
   *     using Mat = Eigen::Matrix<double, 2, 2>;
   *     Mat m = randomize<Mat>(std::index_sequence<>, N {1.0, 0.3}));
   *   \endcode
   *  - One distribution for each matrix element. The following code constructs a 2-by-2 matrix m containing
   *  random values around mean 1.0, 2.0, 3.0, and 4.0 (in row-major order), with standard deviations of
   *  0.3, 0.3, 0.0 (by default, since no s.d. is specified as a parameter), and 0.3:
   *   \code
   *     using N = std::normal_distribution<double>;
   *     auto m = randomize<Eigen::Matrix<double, 2, 2>>(std::index_sequence<0, 1>, N {1.0, 0.3}, N {2.0, 0.3}, 3.0, N {4.0, 0.3})));
   *   \endcode
   *  - One distribution for each row. The following code constructs a 3-by-2 (n) or 2-by-2 (o) matrices
   *  in which elements in each row are selected according to the three (n) or two (o) listed distribution
   *  parameters:
   *   \code
   *     auto n = randomize<Eigen::Matrix<double, 3, 2>>(std::index_sequence<0>, N {1.0, 0.3}, 2.0, N {3.0, 0.3})));
   *     auto o = randomize<Eigen::Matrix<double, 2, 2>>(std::index_sequence<0>, N {1.0, 0.3}, N {2.0, 0.3})));
   *   \endcode
   *   Note that in the case of o, there is an ambiguity as to whether the listed distributions correspond to rows
   *   or columns. In case of such an ambiguity, this function assumes that the parameters correspond to the rows.
   *  - One distribution for each column. The following code constructs 2-by-3 matrix p
   *  in which elements in each column are selected according to the three listed distribution parameters:
   *   \code
   *     auto p = randomize<Eigen::Matrix<double, 2, 3>>(std::index_sequence<1>, N {1.0, 0.3}, 2.0, N {3.0, 0.3})));
   *   \endcode
   * \tparam PatternMatrix A fixed-size matrix
   */
#ifdef __cpp_concepts
  template<indexible PatternMatrix, std::uniform_random_bit_generator random_number_generator = std::mt19937,
    std::size_t...indices, typename...Dists>
  requires
    (not has_dynamic_dimensions<PatternMatrix>) and
    (sizeof...(Dists) == detail::pattern_index_dims<PatternMatrix, indices...>(
      std::make_index_sequence<max_indices_of_v<PatternMatrix>> {})) and
    std::constructible_from<random_number_generator, typename std::random_device::result_type> and
    ((std::is_arithmetic_v<Dists> or requires { typename Dists::result_type; typename Dists::param_type; }) and ...)
#else
  template<typename PatternMatrix, typename random_number_generator = std::mt19937, std::size_t...indices,
    typename...Dists, std::enable_if_t<indexible<PatternMatrix> and (not has_dynamic_dimensions<PatternMatrix>) and
    (sizeof...(Dists) == detail::pattern_index_dims<PatternMatrix, indices...>(
      std::make_index_sequence<max_indices_of_v<PatternMatrix>> {})) and
    std::is_constructible_v<random_number_generator, typename std::random_device::result_type> and
    ((std::is_arithmetic_v<Dists> or detail::is_std_dist<Dists>::value) and ...), int> = 0>
#endif
  constexpr auto
  randomize(std::index_sequence<indices...> seq, Dists&&...dists)
  {
    auto d_tup = get_all_dimensions_of<PatternMatrix>();
    return randomize<PatternMatrix, random_number_generator>(seq, d_tup, std::forward<Dists>(dists)...);
  }


  /**
   * \overload
   * \brief Fill a fixed-sized matrix with random values selected from a random distribution.
   * \details The distributions are allocated to each element of the matrix, according to one of the following options:
   *  - One distribution for all matrix elements. The following example constructs a 2-by-2 matrix (m) in which each
   *  element is a random value selected based on a distribution with mean 1.0 and standard deviation 0.3:
   *   \code
   *     using N = std::normal_distribution<double>;
   *     using Mat = Eigen::Matrix<double, 2, 2>;
   *     Mat m = randomize<Mat>(N {1.0, 0.3}));
   *   \endcode
   * \tparam PatternMatrix A fixed-size matrix
   */
#ifdef __cpp_concepts
  template<indexible PatternMatrix, std::uniform_random_bit_generator random_number_generator = std::mt19937, typename Dist>
  requires
    (not has_dynamic_dimensions<PatternMatrix>) and
    std::constructible_from<random_number_generator, typename std::random_device::result_type> and
    (std::is_arithmetic_v<Dist> or requires { typename Dist::result_type; typename Dist::param_type; })
#else
  template<typename PatternMatrix, typename random_number_generator = std::mt19937, typename Dist,
    std::enable_if_t<indexible<PatternMatrix> and (not has_dynamic_dimensions<PatternMatrix>) and
    std::is_constructible_v<random_number_generator, typename std::random_device::result_type> and
    (std::is_arithmetic_v<Dist> or detail::is_std_dist<Dist>::value), int> = 0>
#endif
  constexpr auto
  randomize(Dist&& dist)
  {
    return randomize<PatternMatrix, random_number_generator>(std::index_sequence<>{}, std::forward<Dist>(dist));
  }


  // -------- //
  //  reduce  //
  // -------- //

  namespace detail
  {
    template<std::size_t I, std::size_t...index, typename Arg>
    constexpr auto get_reduced_index(Arg&& arg)
    {
      if constexpr (((I == index) or ...))
      {
        using T = coefficient_types_of_t<Arg, I>;
        if constexpr (has_uniform_dimension_type<T>) return uniform_dimension_type_of_t<T>{};
        else return Dimensions<1>{};
      }
      else return get_dimensions_of<I>(std::forward<Arg>(arg));
    }


    template<std::size_t...index, typename T, std::size_t...I>
    constexpr auto make_zero_matrix_reduction(T&& t, std::index_sequence<I...>)
    {
      return make_zero_matrix_like<T>(get_reduced_index<I, index...>(std::forward<T>(t))...);
    }


    template<auto constant, std::size_t...index, typename T, std::size_t...I>
    constexpr auto make_constant_matrix_reduction(T&& t, std::index_sequence<I...>)
    {
      return make_constant_matrix_like<T, constant>(get_reduced_index<I, index...>(std::forward<T>(t))...);
    }


    template<auto constant, std::size_t index, typename T, std::size_t...I>
    constexpr auto make_constant_diagonal_matrix_reduction(T&& t, std::index_sequence<I...>)
    {
      // \todo Handle 3+ dimensional constant diagonal tensors
      static_assert(index == 0 or index == 1);
      return make_constant_matrix_like<T, constant>(get_reduced_index<I, index>(std::forward<T>(t))...);
    }


    template<typename T, std::size_t...indices, std::size_t...I>
    constexpr std::size_t count_reduced_dimensions(std::index_sequence<I...>)
    {
      return ([]{
        if constexpr (is_index_match<I, indices...>()) return index_dimension_of_v<T, I>;
        else return 1;
      }() * ...);
    }


    template<std::size_t...indices, typename T, std::size_t...I>
    constexpr std::size_t count_reduced_dimensions(const T& t, std::index_sequence<I...>)
    {
      return ([](const T& t){
        if constexpr (is_index_match<I, indices...>()) return get_index_dimension_of<I>(t);
        else return 1;
      }(t) * ...);
    }


    template<std::size_t dim, typename BinaryFunction, std::size_t...index, typename Scalar>
    constexpr Scalar calc_reduce_constant(Scalar constant)
    {
      if constexpr (dim <= 1)
        return constant;
      else if constexpr (is_plus<BinaryFunction>::value)
        return constant * dim;
      else if constexpr (is_multiplies<BinaryFunction>::value)
        return internal::constexpr_pow(constant, dim);
      else
        return BinaryFunction{}(constant, calc_reduce_constant<dim - 1, BinaryFunction>(constant));
    }


    template<std::size_t...index, typename BinaryFunction, typename Scalar>
    constexpr Scalar calc_reduce_constant(std::size_t dim, const BinaryFunction& b, Scalar constant)
    {
      if (dim <= 1)
        return constant;
      else if constexpr (is_plus<BinaryFunction>::value)
        return constant * dim;
      else if constexpr (is_multiplies<BinaryFunction>::value)
        return std::pow(constant, dim);
      else
        return b(constant, calc_reduce_constant(dim - 1, b, constant));
    }


    template<typename Arg, std::size_t...I>
    constexpr bool has_uniform_reduction_indices(std::index_sequence<I...>)
    {
      return ((has_uniform_dimension_type<coefficient_types_of_t<Arg, I>> or dynamic_dimension<Arg, I>) and ...);
    }


    template<typename BinaryFunction, typename Arg, std::size_t...indices>
    constexpr scalar_type_of_t<Arg>
    reduce_all_indices(const BinaryFunction& b, Arg&& arg, std::index_sequence<indices...>)
    {
      if constexpr (zero_matrix<Arg> and (is_plus<BinaryFunction>::value or is_multiplies<BinaryFunction>::value))
      {
        return 0;
      }
      else if constexpr (constant_matrix<Arg>)
      {
        using Scalar = scalar_type_of_t<Arg>;
        constexpr Scalar c = constant_coefficient_v<Arg>;
        constexpr auto seq = std::make_index_sequence<max_indices_of_v<Arg>> {};
        constexpr bool fixed_reduction_dims = not (dynamic_dimension<Arg, indices> or ...);

        if constexpr (fixed_reduction_dims and is_constexpr_n_ary_function<BinaryFunction, Scalar, Scalar>::value)
        {
          constexpr std::size_t dim = count_reduced_dimensions<Arg, indices...>(seq);
          return calc_reduce_constant<dim, BinaryFunction, indices...>(c);
        }
        else
        {
          std::size_t dim = count_reduced_dimensions<indices...>(arg, seq);
          return calc_reduce_constant<indices...>(dim, b, c);
        }
      }
      else
      {
        decltype(auto) red = interface::ArrayOperations<std::decay_t<Arg>>::template reduce<indices...>(b, std::forward<Arg>(arg));
        using Red = decltype(red);

        static_assert(scalar_type<Red> or
          ((index_dimension_of_v<Red, indices> == 1 or index_dimension_of_v<Red, indices> == dynamic_size) and ...));

        if constexpr (scalar_type<Red>)
          return std::forward<Red>(red);
        else if constexpr (element_gettable<Red, decltype(indices)...>)
          return get_element(std::forward<Red>(red), static_cast<decltype(indices)>(0)...);
        else
          return interface::LinearAlgebra<std::decay_t<Red>>::trace(std::forward<Red>(red));
      }
    }

  } // namespace detail


  /**
   * \brief Perform a complete reduction based on an associative binary function, and return a scalar.
   * \details The binary function must be associative. (This is not enforced, but the order of operation is undefined.)
   * \tparam BinaryFunction A binary function invocable with two values of type <code>scalar_type_of_t<Arg></code>.
   * It must be an associative function. Preferably, it should be a constexpr function, and even more preferably,
   * it should be a standard c++ function such as std::plus or std::multiplies.
   * \tparam Arg The tensor
   * \returns A scalar representing a complete reduction.
   */
#ifdef __cpp_concepts
  template<typename BinaryFunction, indexible Arg> requires
    std::is_invocable_r_v<scalar_type_of_t<Arg>, BinaryFunction&&, scalar_type_of_t<Arg>, scalar_type_of_t<Arg>> and
    (detail::has_uniform_reduction_indices<Arg>(std::make_index_sequence<max_indices_of_v<Arg>> {}))
#else
  template<typename BinaryFunction, typename Arg, std::enable_if_t<indexible<Arg> and
    std::is_invocable_r<typename scalar_type_of<Arg>::type, BinaryFunction&&,
      typename scalar_type_of<Arg>::type, typename scalar_type_of<Arg>::type>::value and
    (detail::has_uniform_reduction_indices<Arg>(std::make_index_sequence<max_indices_of_v<Arg>> {})), int> = 0>
#endif
  constexpr decltype(auto)
  reduce(const BinaryFunction& b, Arg&& arg)
  {
    constexpr auto max_indices = max_indices_of_v<Arg>;
    return detail::reduce_all_indices(b, std::forward<Arg>(arg), std::make_index_sequence<max_indices> {});
  }


  /**
   * \overload
   * \brief Perform a partial reduction based on an associative binary function, across one or more indices.
   * \details The binary function must be associative. (This is not enforced, but the order of operation is undefined.)
   * \tparam index an index to be reduced. For example, if the index is 0, the result will have only one row.
   * If the index is 1, the result will have only one column.
   * \tparam indices Other indicesto be reduced. Because the binary function is associative, the order
   * of the indices does not matter.
   * \tparam BinaryFunction A binary function invocable with two values of type <code>scalar_type_of_t<Arg></code>.
   * It must be an associative function. Preferably, it should be a constexpr function, and even more preferably,
   * it should be a standard c++ function such as std::plus or std::multiplies.
   * \tparam Arg The tensor
   * \returns A vector or tensor with reduced dimensions.
   */
#ifdef __cpp_concepts
  template<std::size_t index, std::size_t...indices, typename BinaryFunction, indexible Arg> requires
    ((index < max_indices_of_v<Arg>) and ... and (indices < max_indices_of_v<Arg>)) and
    std::is_invocable_r_v<scalar_type_of_t<Arg>, BinaryFunction&&, scalar_type_of_t<Arg>, scalar_type_of_t<Arg>> and
    (detail::has_uniform_reduction_indices<Arg>(std::index_sequence<index, indices...> {}))
#else
  template<std::size_t index, std::size_t...indices, typename BinaryFunction, typename Arg, std::enable_if_t<
    indexible<Arg> and ((index < max_indices_of<Arg>::value) and ... and (indices < max_indices_of<Arg>::value)) and
    std::is_invocable_r<typename scalar_type_of<Arg>::type, BinaryFunction&&,
      typename scalar_type_of<Arg>::type, typename scalar_type_of<Arg>::type>::value and
    (detail::has_uniform_reduction_indices<Arg>(std::index_sequence<index, indices...> {})), int> = 0>
#endif
  constexpr auto
  reduce(const BinaryFunction& b, Arg&& arg)
  {
    constexpr auto max_indices = max_indices_of_v<Arg>;
    constexpr std::make_index_sequence<max_indices> seq;

    if constexpr (covariance<Arg>)
    {
      using RC = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<coefficient_types_of_t<Arg, 0>>, coefficient_types_of_t<Arg, 0>>;
      using CC = std::conditional_t<((index == 1) or ... or (indices == 1)),
        uniform_dimension_type_of_t<coefficient_types_of_t<Arg, 1>>, coefficient_types_of_t<Arg, 1>>;
      auto m = reduce<index, indices...>(b, to_covariance_nestable(std::forward<Arg>(arg)));
      return Matrix<RC, CC, decltype(m)> {std::move(m)};
    }
    else if constexpr(mean<Arg> and ((index != 0) or ... or (indices != 0)))
    {
      using C = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<coefficient_types_of_t<Arg, 0>>, coefficient_types_of_t<Arg, 0>>;
      auto m = from_euclidean<C>(reduce<index, indices...>(b, nested_matrix(to_euclidean(std::forward<Arg>(arg)))));
      return Mean<C, decltype(m)> {std::move(m)};
    }
    else if constexpr (euclidean_transformed<Arg> and ((index != 0) or ... or (indices != 0)))
    {
      using C = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<coefficient_types_of_t<Arg, 0>>, coefficient_types_of_t<Arg, 0>>;
      auto m = reduce<index, indices...>(b, nested_matrix(std::forward<Arg>(arg)));
      return EuclideanMean<C, decltype(m)> {std::move(m)};
    }
    else if constexpr (typed_matrix<Arg>)
    {
      using RC = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<coefficient_types_of_t<Arg, 0>>, coefficient_types_of_t<Arg, 0>>;
      using CC = std::conditional_t<((index == 1) or ... or (indices == 1)),
        uniform_dimension_type_of_t<coefficient_types_of_t<Arg, 1>>, coefficient_types_of_t<Arg, 1>>;
      auto m = reduce<index, indices...>(b, nested_matrix(std::forward<Arg>(arg)));
      return Matrix<RC, CC, decltype(m)> {std::move(m)};
    }
    else if constexpr (index_dimension_of_v<Arg, index> == 1)
    {
      if constexpr (sizeof...(indices) == 0) return std::forward<Arg>(arg);
      else return reduce<indices...>(b, std::forward<Arg>(arg));
    }
    else if constexpr (zero_matrix<Arg> and (detail::is_plus<BinaryFunction>::value or detail::is_multiplies<BinaryFunction>::value))
    {
      return detail::make_zero_matrix_reduction<index, indices...>(std::forward<Arg>(arg), seq);
    }
    else if constexpr (constant_matrix<Arg>)
    {
      using Scalar = scalar_type_of_t<Arg>;
      constexpr Scalar c_arg = constant_coefficient_v<Arg>;
      constexpr bool fixed_reduction_dims = not (dynamic_dimension<Arg, index> or ... or dynamic_dimension<Arg, indices>);

      if constexpr (fixed_reduction_dims and detail::is_constexpr_n_ary_function<BinaryFunction, Scalar, Scalar>::value)
      {
        constexpr std::size_t dim = detail::count_reduced_dimensions<Arg, index, indices...>(seq);
        constexpr auto c = detail::calc_reduce_constant<dim, BinaryFunction, index, indices...>(c_arg);
# if __cpp_nontype_template_args >= 201911L
        return detail::make_constant_matrix_reduction<c, indices...>(std::forward<Arg>(arg), seq);
# else
        constexpr auto c_integral = static_cast<std::intmax_t>(c);
        if constexpr (are_within_tolerance(c, static_cast<Scalar>(c_integral)))
        {
          return detail::make_constant_matrix_reduction<c_integral, index, indices...>(std::forward<Arg>(arg), seq);
        }
        else
        {
          auto red = detail::make_constant_matrix_reduction<1, index, indices...>(std::forward<Arg>(arg), seq);
          return make_self_contained(c * to_native_matrix<Arg>(std::move(red)));
        }
# endif
      }
      else
      {
        std::size_t dim = detail::count_reduced_dimensions<index, indices...>(arg, seq);
        auto c = detail::calc_reduce_constant<index, indices...>(dim, b, c_arg);
        auto red = detail::make_constant_matrix_reduction<1, index, indices...>(std::forward<Arg>(arg), seq);
        return make_self_contained(c * to_native_matrix<Arg>(std::move(red)));
      }
    }
    //else if constexpr (constant_diagonal_matrix<Arg>)
    //{
    //  return const_diagonal_reduce<indices...>(b, std::forward<Arg>(arg), seq);
    //}
    else
    {
      return interface::ArrayOperations<std::decay_t<Arg>>::template reduce<index, indices...>(b, std::forward<Arg>(arg));
    }
  }


  // ---------------- //
  //  average_reduce  //
  // ---------------- //

  /**
   * \brief Perform a complete reduction by taking the average along all indices and returning a scalar value.
   * \returns A scalar representing the average of all components.
   */
#ifdef __cpp_concepts
  template<indexible Arg> requires
    (detail::has_uniform_reduction_indices<Arg>(std::make_index_sequence<max_indices_of_v<Arg>> {}))
#else
  template<typename Arg, std::enable_if_t<indexible<Arg> and
    detail::has_uniform_reduction_indices<Arg>(std::make_index_sequence<max_indices_of_v<Arg>> {}), int> = 0>
#endif
  constexpr decltype(auto)
  average_reduce(Arg&& arg) noexcept
  {
    if constexpr (zero_matrix<Arg>)
      return 0;
    else if constexpr (constant_matrix<Arg>)
      return constant_coefficient_v<Arg>;
    else
      return reduce(std::plus<scalar_type_of_t<Arg>> {}, std::forward<Arg>(arg)) / (
        std::apply([](const auto&...d){ return (get_dimension_size_of(d) * ...); }, get_all_dimensions_of(arg)));
  }


  /**
   * \overload
   * \brief Perform a partial reduction by taking the average along one or more indices.
   * \tparam index an index to be reduced. For example, if the index is 0, the result will have only one row.
   * If the index is 1, the result will have only one column.
   * \tparam indices Other indicesto be reduced. Because the binary function is associative, the order
   * of the indices does not matter.
   * \returns A vector or tensor with reduced dimensions.
   */
#ifdef __cpp_concepts
  template<std::size_t index, std::size_t...indices, indexible Arg> requires
    ((index < max_indices_of_v<Arg>) and ... and (indices < max_indices_of_v<Arg>)) and
    (detail::has_uniform_reduction_indices<Arg>(std::index_sequence<index, indices...> {}))
#else
  template<std::size_t index, std::size_t...indices, typename Arg, std::enable_if_t<indexible<Arg> and
    ((index < max_indices_of_v<Arg>) and ... and (indices < max_indices_of_v<Arg>)) and
    (detail::has_uniform_reduction_indices<Arg>(std::index_sequence<index, indices...> {})), int> = 0>
#endif
  constexpr auto
  average_reduce(Arg&& arg) noexcept
  {
    using Scalar = scalar_type_of_t<Arg>;
    constexpr auto max_indices = max_indices_of_v<Arg>;
    constexpr std::make_index_sequence<max_indices> seq;

    if constexpr (covariance<Arg>)
    {
      using RC = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<coefficient_types_of_t<Arg, 0>>, coefficient_types_of_t<Arg, 0>>;
      using CC = std::conditional_t<((index == 1) or ... or (indices == 1)),
        uniform_dimension_type_of_t<coefficient_types_of_t<Arg, 1>>, coefficient_types_of_t<Arg, 1>>;
      auto m = average_reduce<index, indices...>(to_covariance_nestable(std::forward<Arg>(arg)));
      return Matrix<RC, CC, decltype(m)> {std::move(m)};
    }
    else if constexpr(mean<Arg> and ((index != 0) or ... or (indices != 0)))
    {
      using C = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<coefficient_types_of_t<Arg, 0>>, coefficient_types_of_t<Arg, 0>>;
      auto m = from_euclidean<C>(average_reduce<index, indices...>(nested_matrix(to_euclidean(std::forward<Arg>(arg)))));
      return Mean<C, decltype(m)> {std::move(m)};
    }
    else if constexpr (euclidean_transformed<Arg> and ((index != 0) or ... or (indices != 0)))
    {
      using C = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<coefficient_types_of_t<Arg, 0>>, coefficient_types_of_t<Arg, 0>>;
      auto m = average_reduce<index, indices...>(nested_matrix(std::forward<Arg>(arg)));
      return EuclideanMean<C, decltype(m)> {std::move(m)};
    }
    else if constexpr (typed_matrix<Arg>)
    {
      using RC = std::conditional_t<((index == 0) or ... or (indices == 0)),
        uniform_dimension_type_of_t<coefficient_types_of_t<Arg, 0>>, coefficient_types_of_t<Arg, 0>>;
      using CC = std::conditional_t<((index == 1) or ... or (indices == 1)),
        uniform_dimension_type_of_t<coefficient_types_of_t<Arg, 1>>, coefficient_types_of_t<Arg, 1>>;
      auto m = average_reduce<index, indices...>(nested_matrix(std::forward<Arg>(arg)));
      return Matrix<RC, CC, decltype(m)> {std::move(m)};
    }
    else if constexpr (index_dimension_of_v<Arg, index> == 1)
    {
      if constexpr (sizeof...(indices) == 0) return std::forward<Arg>(arg);
      else return average_reduce<indices...>(std::forward<Arg>(arg));
    }
    else if constexpr (zero_matrix<Arg>)
    {
      return detail::make_zero_matrix_reduction<index, indices...>(std::forward<Arg>(arg), seq);
    }
    else if constexpr (constant_matrix<Arg>)
    {
      constexpr Scalar c = constant_coefficient_v<Arg>;
# if __cpp_nontype_template_args >= 201911L
      return detail::make_constant_matrix_reduction<c, index, indices...>(std::forward<Arg>(arg), seq);
# else
      constexpr auto c_integral = static_cast<std::intmax_t>(c);
      if constexpr (are_within_tolerance(c, static_cast<Scalar>(c_integral)))
      {
        return detail::make_constant_matrix_reduction<c_integral, index, indices...>(std::forward<Arg>(arg), seq);
      }
      else
      {
        auto red = detail::make_constant_matrix_reduction<1, index, indices...>(std::forward<Arg>(arg), seq);
        return make_self_contained(c * to_native_matrix<Arg>(red));
      }
# endif
    }
    else if constexpr (constant_diagonal_matrix<Arg>)
    {
      if constexpr (not dynamic_dimension<Arg, 0>)
      {
        constexpr auto c = static_cast<Scalar>(constant_diagonal_coefficient_v<Arg>) / index_dimension_of_v<Arg, 0>;
# if __cpp_nontype_template_args >= 201911L
        return average_reduce<indices...>(detail::make_constant_diagonal_matrix_reduction<c, index>(std::forward<Arg>(arg), seq));
# else
        constexpr auto c_integral = static_cast<std::intmax_t>(c);
        if constexpr (are_within_tolerance(c, static_cast<Scalar>(c_integral)))
        {
          auto ret = detail::make_constant_diagonal_matrix_reduction<c_integral, index>(std::forward<Arg>(arg), seq);
          if constexpr (sizeof...(indices) == 0) return ret;
          else return average_reduce<indices...>(std::move(ret));
        }
        else
        {
          auto ret = detail::make_constant_diagonal_matrix_reduction<1, index>(std::forward<Arg>(arg), seq);
          if constexpr (sizeof...(indices) == 0) return make_self_contained(c * std::move(ret));
          else return make_self_contained(c * to_native_matrix<Arg>(average_reduce<indices...>(std::move(ret))));
        }
# endif
      }
      else
      {
        auto c = static_cast<Scalar>(constant_diagonal_coefficient_v<Arg>) / get_index_dimension_of<0>(arg);
        auto ret = detail::make_constant_diagonal_matrix_reduction<1, index>(std::forward<Arg>(arg), seq);
        if constexpr (sizeof...(indices) == 0) return make_self_contained(c * std::move(ret));
        else return make_self_contained(c * average_reduce<indices...>(std::move(ret)));
      }
    }
    else
    {
      return make_self_contained(reduce<index, indices...>(std::plus<Scalar> {}, std::forward<Arg>(arg)) /
        (get_index_dimension_of<index>(arg) * ... * get_index_dimension_of<indices>(arg)));
    }
  }


} // namespace OpenKalman

#endif //OPENKALMAN_ARRAY_FUNCTIONS_HPP
