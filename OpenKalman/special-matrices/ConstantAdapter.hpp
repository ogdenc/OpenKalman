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
 * \brief Definitions for ConstantAdapter
 */

#ifndef OPENKALMAN_CONSTANTADAPTER_HPP
#define OPENKALMAN_CONSTANTADAPTER_HPP

#include <type_traits>

namespace OpenKalman
{
  // ----------------- //
  //  ConstantAdapter  //
  // ----------------- //

#ifdef __cpp_concepts
  template<indexible PatternMatrix, auto...constant>
  requires std::constructible_from<scalar_type_of_t<PatternMatrix>, decltype(constant)...>
#else
  template<typename PatternMatrix, auto...constant>
#endif
  struct ConstantAdapter : internal::library_base<ConstantAdapter<PatternMatrix, constant...>, PatternMatrix>
  {

  private:

#ifndef __cpp_concepts
    static_assert(std::is_constructible_v<scalar_type_of_t<PatternMatrix>, decltype(constant)...>);
#endif

    using MyDimensions = decltype(get_all_dimensions_of(std::declval<PatternMatrix>()));


    template<std::size_t N = 0>
    constexpr auto make_all_dimensions_tuple() { return std::tuple {}; }


    template<std::size_t N = 0, typename D, typename...Ds>
    constexpr auto make_all_dimensions_tuple(D&& d, Ds&&...ds)
    {
      using E = std::tuple_element_t<N, MyDimensions>;
      if constexpr (fixed_index_descriptor<E>)
      {
        auto e = std::get<N>(my_dimensions);
        if constexpr (fixed_index_descriptor<D>) static_assert(equivalent_to<D, E>);
        else if (e != d) throw std::invalid_argument {"Invalid argument for index descriptors of a constant matrix"};
        return std::tuple_cat(std::forward_as_tuple(std::move(e)),
          make_all_dimensions_tuple<N + 1>(std::forward<Ds>(ds)...));
      }
      else
      {
        return std::tuple_cat(std::forward_as_tuple(std::forward<D>(d)),
          make_all_dimensions_tuple<N + 1>(std::forward<Ds>(ds)...));
      }
    }


  public:

    /**
     * \brief Construct a ConstantAdapter using a full set of index descriptors.
     * \tparam D A set of \ref index_descriptor "index_descriptors" corresponding to class template parameters Ds.
     * \details Each D must be a constructor argument for Ds.
     * For example, the following construct a 2-by-3 constant matrix of value 5:
     * \code
     * ConstantMatrix<double, 5, std::size_t, std::size_t>(2, 3)
     * ConstantMatrix<double, 5, Dimensions<2>, std::size_t>(std::integral_constant<int, 2>{}, 3)
     * ConstantMatrix<double, 5, int, std::integral_constant<int, 3>>(2, std::integral_constant<int, 3>{})
     * ConstantMatrix<double, 5, std::integral_constant<std::size_t, 2>, Dimensions<3>>(std::integral_constant<int, 2>{}, Dimensions<3>{})
     * \endcode
     */
#ifdef __cpp_concepts
    template<index_descriptor...Ds> requires (sizeof...(Ds) == std::tuple_size_v<MyDimensions>)
#else
    template<typename...Ds, std::enable_if_t<(index_descriptor<Ds> and ...) and
      (sizeof...(Ds) == std::tuple_size_v<MyDimensions>), int> = 0>
#endif
    constexpr ConstantAdapter(Ds&&...ds) : my_dimensions {make_all_dimensions_tuple(std::forward<Ds>(ds)...)} {}


  private:

    template<std::size_t N = 0>
    constexpr auto make_dynamic_dimensions_tuple()
    {
      if constexpr (N < max_indices_of_v<PatternMatrix>)
        return std::tuple_cat(std::forward_as_tuple(std::get<N>(my_dimensions)), make_dynamic_dimensions_tuple<N + 1>());
      else
        return std::tuple {};
    }


    template<std::size_t N = 0, typename D, typename...Ds>
    constexpr auto make_dynamic_dimensions_tuple(D&& d, Ds&&...ds)
    {
      if constexpr (dynamic_index_descriptor<std::tuple_element_t<N, MyDimensions>>)
        return std::tuple_cat(std::forward_as_tuple(std::forward<D>(d)),
          make_dynamic_dimensions_tuple<N + 1>(std::forward<Ds>(ds)...));
      else
        return std::tuple_cat(std::forward_as_tuple(std::get<N>(my_dimensions)),
          make_dynamic_dimensions_tuple<N + 1>(std::forward<D>(d), std::forward<Ds>(ds)...));
    }


  public:

    /**
     * \brief Construct a ConstantAdapter using any applicable dynamic index descriptors.
     * \tparam Ds A set of \ref dynamic_index_descriptor "dynamic index descriptors" corresponding to each of
     * class template parameter Ds that is dynamic, in order of Ds. This list should omit
     * any \ref fixed_index_descriptor "fixed index descriptors".
     * \details If PatternMatrix has no dynamic dimensions, this is a default constructor.
     * The constructor can take a number of arguments representing the number of dynamic dimensions.
     * For example, the following construct a 2-by-3 constant matrix of value 5:
     * \code
     * ConstantAdapter<eigen_matrix_t<double, dynamic_size, dynamic_size>, 5>(2, 3) // Dynamic rows and columns.
     * ConstantAdapter<eigen_matrix_t<double, 2, dynamic_size>, 5>(3) // Fixed rows and dynamic columns.
     * ConstantAdapter<eigen_matrix_t<double, dynamic_size, 3>, 5>(2) // Dynamic rows and fixed columns.
     * ConstantAdapter<eigen_matrix_t<double, 2, 3>, 5>() // Fixed rows and columns.
     * \endcode
     */
#ifdef __cpp_concepts
    template<dynamic_index_descriptor...Ds> requires (sizeof...(Ds) < std::tuple_size_v<MyDimensions>) and
      (sizeof...(Ds) == number_of_dynamic_indices_v<PatternMatrix>)
#else
    template<typename...Ds, std::enable_if_t<(dynamic_index_descriptor<Ds> and ...) and
      (sizeof...(Ds) < std::tuple_size_v<MyDimensions>) and sizeof...(Ds) == number_of_dynamic_indices_v<PatternMatrix>, int> = 0>
#endif
    constexpr ConstantAdapter(Ds&&...ds) : my_dimensions {make_dynamic_dimensions_tuple(std::forward<Ds>(ds)...)} {}


  private:

    template<std::size_t N = 0, typename Arg>
    constexpr auto make_dimensions_tuple(const Arg& arg)
    {
      if constexpr (N < std::tuple_size_v<MyDimensions>)
      {
        using E = std::tuple_element_t<N, MyDimensions>;
        if constexpr (fixed_index_descriptor<E>)
        {
          auto e = std::get<N>(my_dimensions);
          using D = coefficient_types_of_t<Arg, N>;
          if constexpr (fixed_index_descriptor<D>) static_assert(equivalent_to<D, E>);
          else if (e != get_dimensions_of<N>(arg))
            throw std::invalid_argument {"Invalid argument for index descriptors of a constant matrix"};
          return std::tuple_cat(std::tuple{std::move(e)}, make_dimensions_tuple<N + 1>(arg));
        }
        else
        {
          return std::tuple_cat(std::tuple{get_dimensions_of<N>(arg)}, make_dimensions_tuple<N + 1>(arg));
        }
      }
      else
      {
        return std::tuple {};
      }
    }


  public:

    /**
     * \brief Construct a ConstantMatrix from another \ref constant_matrix.
     */
#ifdef __cpp_concepts
    template<constant_matrix Arg> requires (not std::derived_from<Arg, ConstantAdapter>) and
      (constant_coefficient_v<Arg> == scalar_type_of_t<PatternMatrix> {constant...}) and
      maybe_has_same_shape_as<Arg, PatternMatrix>
#else
    template<typename Arg, std::enable_if_t<constant_matrix<Arg> and (not std::is_base_of_v<ConstantAdapter, Arg>) and
      (constant_coefficient<Arg>::value == scalar_type_of_t<PatternMatrix> {constant...}) and
      maybe_has_same_shape_as<Arg, PatternMatrix>, int> = 0>
#endif
    constexpr ConstantAdapter(const Arg& arg) : my_dimensions {make_dimensions_tuple(arg)} {}


  private:

    template<std::size_t N = 0, typename Arg>
    constexpr bool index_descriptors_match_this(const Arg& arg)
    {
      if constexpr (N < std::tuple_size_v<MyDimensions>)
      {
        using E = std::tuple_element_t<N, MyDimensions>;
        if constexpr (fixed_index_descriptor<E>)
          return std::get<N>(my_dimensions) == get_dimensions_of<N>(arg) and index_descriptors_match_this<N + 1>(arg);
        else
          return std::is_constructible_v<E, coefficient_types_of_t<Arg, N>> and index_descriptors_match_this<N + 1>(arg);
      }
      else return true;
    }


  public:

    /**
     * \brief Assign from another compatible \ref constant_matrix.
     */
#ifdef __cpp_concepts
    template<constant_matrix Arg> requires (not std::derived_from<Arg, ConstantAdapter>) and
      (constant_coefficient_v<Arg> == scalar_type_of_t<PatternMatrix> {constant...}) and
      maybe_has_same_shape_as<Arg, PatternMatrix>
#else
    template<typename Arg, std::enable_if_t<constant_matrix<Arg> and (not std::is_base_of_v<ConstantAdapter, Arg>) and
      (constant_coefficient<Arg>::value == scalar_type_of_t<PatternMatrix> {constant...}) and
      maybe_has_same_shape_as<Arg, PatternMatrix>, int> = 0>
#endif
    constexpr auto& operator=(Arg&& arg)
    {
      if constexpr (not has_same_shape_as<Arg, PatternMatrix>) if (not index_descriptors_match_this(arg))
        throw std::invalid_argument {"Argument to ConstantAdapter assignment operator has non-matching index descriptors."};
      return *this;
    }


    /**
     * \brief Comparison operator
     */
#ifdef __cpp_concepts
    template<indexible Arg>
#else
    template<typename Arg, std::enable_if_t<indexible<Arg>, int> = 0>
#endif
    constexpr bool operator==(const Arg& arg) const
    {
      if constexpr (not maybe_has_same_shape_as<Arg, PatternMatrix>)
      {
        return false;
      }
      else if constexpr (constant_matrix<Arg>)
      {
        if constexpr (constant_coefficient_v<Arg> == scalar_type_of_t<PatternMatrix> {constant...})
          return get_index_descriptors_match(*this, arg);
        else
          return false;
      }
      else
      {
        auto c = to_native_matrix<PatternMatrix>(*this);
        static_assert(not std::is_same_v<decltype(c), ConstantAdapter>,
          "interface::EquivalentDenseWritableMatrix<PatternMatrix>::to_native_matrix(*this) must define an object within the library of Arg");
        return std::move(c) == arg;
      }
    }


    constexpr const scalar_type_of_t<PatternMatrix> value() const
    {
      return {constant...};
    }


#ifndef __cpp_impl_three_way_comparison
    /**
     * \overload
     */
#ifdef __cpp_concepts
    template<indexible Arg> (not constant_adapter<Arg>)
#else
    template<typename Arg, std::enable_if_t<indexible<Arg> and not constant_adapter<Arg>, int> = 0>
#endif
    friend constexpr bool operator==(const Arg& arg, const ConstantAdapter& c)
    {
      if constexpr (not maybe_has_same_shape_as<Arg, PatternMatrix>)
        return false;
      else if constexpr (constant_matrix<Arg>)
        return constant_coefficient_v<Arg> == scalar_type_of_t<PatternMatrix>{constant...} and get_index_descriptors_match(arg, c);
      else
      {
        auto new_c = to_native_matrix<Arg>(c);
        static_assert(not std::is_same_v<decltype(new_c), ConstantAdapter>,
          "interface::EquivalentDenseWritableMatrix<Arg>::to_native_matrix(c) must define an object within the library of Arg");
        return arg == std::move(new_c);
      }
    }


#ifdef __cpp_concepts
    template<indexible Arg>
#else
    template<typename Arg, std::enable_if_t<indexible<Arg>, int> = 0>
#endif
    constexpr bool operator!=(const Arg& arg) const { return not (*this == arg); }


#ifdef __cpp_concepts
    template<indexible Arg> (not std::same_as<Arg, ConstantAdapter>)
#else
    template<typename Arg, std::enable_if_t<indexible<Arg> and not std::is_same_v<Arg, ConstantAdapter>, int> = 0>
#endif
    friend constexpr bool operator!=(const Arg& arg, const ConstantAdapter& c) { return not (arg == c); }
#endif


    /**
     * \brief Element accessor.
     * \note Does not do any runtime bounds checking.
     * \param is The indices
     * \return The element corresponding to the indices (always the constant).
     */
#ifdef __cpp_concepts
    template<index_value...Is> requires (sizeof...(Is) <= max_indices_of_v<PatternMatrix>)
#else
    template<typename...Is, std::enable_if_t<
      (index_value<Is> and ...) and (sizeof...(Is) <= max_indices_of_v<PatternMatrix>), int> = 0>
#endif
    constexpr scalar_type_of_t<PatternMatrix>
    operator()(Is...is) const
    {
      return {constant...};
    }


    /**
     * \brief Element accessor (single index).
     * \note Does not do any runtime bounds checking.
     * \param is The index
     * \return The element corresponding to the indices (always the constant).
     */
#ifdef __cpp_concepts
    template<index_value Is>
#else
    template<typename Is, std::enable_if_t<index_value<Is>, int> = 0>
#endif
    constexpr scalar_type_of_t<PatternMatrix>
    operator[](Is is) const
    {
      return {constant...};
    }

  protected:

    MyDimensions my_dimensions;


#ifdef __cpp_concepts
    template<typename T, std::size_t N> friend struct interface::IndexTraits;
    template<typename T, std::size_t N> friend struct interface::CoordinateSystemTraits;
#else
    template<typename T, std::size_t N, typename Enable> friend struct interface::IndexTraits;
    template<typename T, std::size_t N, typename Enable> friend struct interface::CoordinateSystemTraits;
#endif

  };


  // ------------------ //
  //  Deduction guides  //
  // ------------------ //

#ifdef __cpp_concepts
  template<constant_matrix Arg> requires (not constant_adapter<Arg>) and
    requires { typename ConstantAdapter<std::decay_t<Arg>, constant_coefficient_v<Arg>>; }
  ConstantAdapter(Arg&&) -> ConstantAdapter<std::decay_t<Arg>, constant_coefficient_v<Arg>>;

  template<constant_matrix Arg> requires (not constant_adapter<Arg>) and
    (not requires { typename ConstantAdapter<std::decay_t<Arg>, constant_coefficient_v<Arg>>; }) and
    (are_within_tolerance(constant_coefficient_v<Arg>, static_cast<std::intmax_t>(constant_coefficient_v<Arg>)))
  ConstantAdapter(Arg&&) -> ConstantAdapter<std::decay_t<Arg>, static_cast<std::intmax_t>(constant_coefficient_v<Arg>)>;
#else
  template<typename Arg, std::enable_if_t<constant_matrix<Arg> and (not constant_adapter<Arg>) and
    std::is_integral_v<typename scalar_type_of<Arg>::type>, int> = 0>
  ConstantAdapter(Arg&&) -> ConstantAdapter<std::decay_t<Arg>, constant_coefficient_v<Arg>>;

  namespace detail
  {
    template<typename T, typename = void>
    struct rounds_to_integral : std::false_type {};

    template<typename T>
    struct rounds_to_integral<T, std::enable_if_t<
      are_within_tolerance(constant_coefficient<T>::value, static_cast<std::intmax_t>(constant_coefficient<T>::value))>>
      : std::true_type {};
  }

  template<typename Arg, std::enable_if_t<constant_matrix<Arg> and (not constant_adapter<Arg>) and
    (not std::is_integral_v<typename scalar_type_of<Arg>::type>) and (detail::rounds_to_integral<Arg>::value), int> = 0>
  ConstantAdapter(Arg&&) -> ConstantAdapter<std::decay_t<Arg>, static_cast<std::intmax_t>(constant_coefficient_v<Arg>)>;
#endif


  // ------------ //
  //  Interfaces  //
  // ------------ //

  namespace interface
  {
    template<typename PatternMatrix, auto...constant>
    struct IndexibleObjectTraits<ConstantAdapter<PatternMatrix, constant...>>
    {
      static constexpr std::size_t max_indices = max_indices_of_v<PatternMatrix>;
      using scalar_type = scalar_type_of_t<PatternMatrix>;
    };


    template<typename PatternMatrix, auto...constant, std::size_t N>
    struct IndexTraits<ConstantAdapter<PatternMatrix, constant...>, N>
    {
      static constexpr std::size_t dimension = index_dimension_of_v<PatternMatrix, N>;

      template<typename Arg>
      static constexpr std::size_t dimension_at_runtime(const Arg& arg)
      {
        return get_dimension_size_of(std::get<N>(arg.my_dimensions));
      }
    };


    template<typename PatternMatrix, auto...constant, std::size_t N>
    struct CoordinateSystemTraits<ConstantAdapter<PatternMatrix, constant...>, N>
    {
      using coordinate_system_types = coefficient_types_of_t<PatternMatrix, N>;

      template<typename Arg>
      static constexpr auto coordinate_system_types_at_runtime(Arg&& arg)
      {
        return std::get<N>(std::forward<Arg>(arg).my_dimensions);
      }
    };


    template<typename PatternMatrix, auto...constant, typename...I>
#ifdef __cpp_concepts
    struct GetElement<ConstantAdapter<PatternMatrix, constant...>, I...>
#else
    struct GetElement<ConstantAdapter<PatternMatrix, constant...>, void, I...>
#endif
    {
      template<typename Arg>
      static constexpr auto get(Arg&& arg, I...) { return constant_coefficient_v<Arg>; }
    };


    // No SetElement defined  because ConstantAdapter is not writable.


    template<typename PatternMatrix, auto...constant, typename Scalar>
    struct EquivalentDenseWritableMatrix<ConstantAdapter<PatternMatrix, constant...>, Scalar>
    {
      static constexpr bool is_writable = false;

      template<typename...D>
      static auto make_default(D&&...d)
      {
        return make_default_dense_writable_matrix_like<PatternMatrix, Scalar>(std::forward<D>(d)...);
      }

      template<typename Arg>
      static decltype(auto) to_native_matrix(Arg&& arg)
      {
        return OpenKalman::to_native_matrix<PatternMatrix>(std::forward<Arg>(arg));
      }

      template<typename Arg>
      static decltype(auto) convert(Arg&& arg)
      {
        return make_dense_writable_matrix_from(OpenKalman::to_native_matrix<PatternMatrix>(std::forward<Arg>(arg)));
      }
    };


    template<typename PatternMatrix, auto...constant>
    struct Dependencies<ConstantAdapter<PatternMatrix, constant...>>
    {
      static constexpr bool has_runtime_parameters = has_dynamic_dimensions<PatternMatrix>;
      using type = std::tuple<>;
    };


    template<typename PatternMatrix, auto...c, typename Scalar>
    struct SingleConstantMatrixTraits<ConstantAdapter<PatternMatrix, c...>, Scalar>
    {
      template<typename...D>
      static constexpr auto make_zero_matrix(D&&...d)
      {
        return make_zero_matrix_like<PatternMatrix, Scalar>(std::forward<D>(d)...);
      }


      template<auto...constant, typename...D>
      static constexpr auto make_constant_matrix(D&&...d)
      {
        return make_constant_matrix_like<PatternMatrix, Scalar, constant...>(std::forward<D>(d)...);
      }


      template<typename S, typename...D>
      static constexpr auto make_runtime_constant(S&& s, D&&...d)
      {
        return make_constant_matrix_like<PatternMatrix>(std::forward<S>(s), std::forward<D>(d)...);
      }
    };


    template<typename PatternMatrix, auto c, auto...cs>
    struct SingleConstant<ConstantAdapter<PatternMatrix, c, cs...>>
    {
    private:

      struct C
      {
        using value_type = scalar_type_of_t<PatternMatrix>;
        static constexpr value_type value {c, cs...};
        static constexpr Likelihood status = Likelihood::definitely;
        constexpr operator value_type() const noexcept { return value; }
        constexpr value_type operator()() const noexcept { return value; }
      };

    public:

      const ConstantAdapter<PatternMatrix, c, cs...>& xpr;

      static constexpr auto get_constant()
      {
        return C{};
      }
    };


    template<typename PatternMatrix>
    struct SingleConstant<ConstantAdapter<PatternMatrix>>
    {
      const ConstantAdapter<PatternMatrix>& xpr;

      constexpr auto get_constant()
      {
        return xpr.value();
      }
    };


    template<typename PatternMatrix, auto...constant, typename Scalar>
    struct SingleConstantDiagonalMatrixTraits<ConstantAdapter<PatternMatrix, constant...>, Scalar>
    {
      template<typename D>
      static constexpr auto make_identity_matrix(D&& d)
      {
        return make_identity_matrix_like<PatternMatrix, Scalar>(std::forward<D>(d));
      }
    };


    // SingleConstantDiagonal not necessary because SingleConstant is defined
    // DiagonalTraits not necessary because SingleConstant is defined
    // TriangularTraits not necessary because SingleConstant is defined
    // HermitianTraits not necessary because SingleConstant is defined


    template<typename PatternMatrix, auto...constant>
    struct ArrayOperations<ConstantAdapter<PatternMatrix, constant...>>
    {
      template<typename...Ds, typename Op, typename...Args>
      static constexpr decltype(auto)
      n_ary_operation(const std::tuple<Ds...>& d_tup, Op&& op, Args&&...args)
      {
        return ArrayOperations<PatternMatrix>::template n_ary_operation(d_tup, std::forward<Op>(op), std::forward<Args>(args)...);
      }

      template<typename...Ds, typename Op, typename...Args>
      static auto n_ary_operation_with_indices(const std::tuple<Ds...>& d_tup, Op&& op, Args&&...args)
      {
        return ArrayOperations<PatternMatrix>::template n_ary_operation_with_indices(d_tup, std::forward<Op>(op), std::forward<Args>(args)...);
      }

      template<std::size_t...indices, typename BinaryFunction, typename Arg>
      static constexpr decltype(auto)
      reduce(BinaryFunction&& b, Arg&& arg)
      {
        return ArrayOperations<PatternMatrix>::template reduce<indices...>(std::forward<BinaryFunction>(b), std::forward<Arg>(arg));
      }
    };


    // No struct Conversions because to_diagonal and diagonal_of are handled by general functions.
    // No struct Subsets defined because get-block operation is handled by get_block function, and there is no set-block operation.
    // No struct ModularTransformationTraits. Relying on default definition of that trait.


    template<typename PatternMatrix, auto...constant>
    struct LinearAlgebra<ConstantAdapter<PatternMatrix, constant...>>
    {
      // conjugate is not necessary because it is handled by the general conjugate function.
      // transpose is not necessary because it is handled by the general transpose function.
      // adjoint is not necessary because it is handled by the general adjoint function.
      // determinant is not necessary because it is handled by the general determinant function.

      template<typename A, typename B>
      static constexpr auto sum(A&& a, B&& b)
      {
        return LinearAlgebra<PatternMatrix>::sum(std::forward<A>(a), std::forward<B>(b));
      }

      template<typename A, typename B>
      static constexpr auto contract(A&& a, B&& b)
      {
        return LinearAlgebra<PatternMatrix>::contract(std::forward<A>(a), std::forward<B>(b));
      }

      // contract_in_place is not necessary because the argument will not be writable.
      // cholesky_factor is not necessary because it is handled by the general cholesky_factor function.

      template<TriangleType t, typename A, typename U, typename Alpha>
      static decltype(auto) rank_update_self_adjoint(A&& a, U&& u, const Alpha alpha)
      {
        using Trait = interface::LinearAlgebra<PatternMatrix>;
        return Trait::template rank_update_self_adjoint<t>(std::forward<A>(a), std::forward<U>(u), alpha);
      }

      // rank_update_triangular is not necessary because it is handled by the general rank_update_triangular function.
      // solve is not necessary because it is handled by the general solve function.
      // LQ_decomposition is not necessary because it is handled by the general LQ_decomposition function.
      // QR_decomposition is not necessary because it is handled by the general QR_decomposition function.

    };

  } // namespace interface


  // --------------------------- //
  //        MatrixTraits         //
  // --------------------------- //

  template<typename PatternMatrix, auto...constant>
  struct MatrixTraits<ConstantAdapter<PatternMatrix, constant...>>
  {
  private:

    using Matrix = ConstantAdapter<PatternMatrix, constant...>;

  public:

    template<typename Derived>
    using MatrixBaseFrom = typename MatrixTraits<std::decay_t<PatternMatrix>>::template MatrixBaseFrom<Derived>;

    template<TriangleType storage_triangle = TriangleType::diagonal>
    using SelfAdjointMatrixFrom = SelfAdjointMatrix<Matrix, storage_triangle>;

    template<TriangleType triangle_type = TriangleType::diagonal>
    using TriangularMatrixFrom = TriangularMatrix<Matrix, triangle_type>;

    template<std::size_t dim = row_dimension_of_v<PatternMatrix>>
    using DiagonalMatrixFrom = DiagonalMatrix<ConstantAdapter<
      untyped_dense_writable_matrix_t<PatternMatrix, scalar_type_of_t<PatternMatrix>, dim, 1>, constant...>>;

  };


} // namespace OpenKalman


#endif //OPENKALMAN_CONSTANTADAPTER_HPP
