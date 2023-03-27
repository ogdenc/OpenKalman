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
  template<indexible PatternMatrix, scalar_constant Scalar, auto...constant>
    requires (sizeof...(constant) == 0) or requires { Scalar {constant...}; }
#else
  template<typename PatternMatrix, typename Scalar, auto...constant>
#endif
  struct ConstantAdapter : internal::library_base<ConstantAdapter<PatternMatrix, Scalar, constant...>, PatternMatrix>
  {

  private:

#ifndef __cpp_concepts
    static_assert(indexible<PatternMatrix>);
    static_assert(scalar_constant<Scalar>);
    static_assert(sizeof...(constant) == 0 or std::is_constructible_v<Scalar, decltype(constant)...>);
#endif

    using MyConstant = std::conditional_t<sizeof...(constant) == 0, Scalar, internal::KnownScalarConstant<Scalar, constant...>>;

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
     * \brief Construct a ConstantAdapter, whose value is known at compile time, using a full set of index descriptors.
     * \tparam D A set of \ref index_descriptor "index_descriptors" corresponding to class template parameters Ds.
     * \details Each D must be a constructor argument for Ds.
     * For example, the following construct a 2-by-3 constant matrix of value 5:
     * \code
     * ConstantAdapter<eigen_matrix_t<double, 2, 3>, 5>(2, std::integral_constant<int, 3>{})
     * ConstantAdapter<eigen_matrix_t<double, Eigen::Dynamic, 3>, 5>(2, 3)
     * ConstantAdapter<eigen_matrix_t<double, Eigen::Dynamic, Eigen::Dynamic>, 5>(2, 3)
     * \endcode
     */
#ifdef __cpp_concepts
    template<index_descriptor...Ds> requires (sizeof...(Ds) > 0) and
      scalar_constant<MyConstant, CompileTimeStatus::known> and compatible_with_index_descriptors<PatternMatrix, Ds...>
#else
    template<typename...Ds, std::enable_if_t<(index_descriptor<Ds> and ...) and (sizeof...(Ds) > 0) and
      scalar_constant<MyConstant, CompileTimeStatus::known> and compatible_with_index_descriptors<PatternMatrix, Ds...>, int> = 0>
#endif
    explicit constexpr ConstantAdapter(Ds&&...ds) : my_dimensions {make_all_dimensions_tuple(std::forward<Ds>(ds)...)} {}


    /**
     * \overload
     * \brief Same as above, except also specifying a \ref scalar_constant.
     * \details For example, the following construct a 2-by-3 constant matrix of value 5:
     * \code
     * ConstantAdapter<eigen_matrix_t<double, 2, 3>, 5>(std::integral_constant<int, 5>{}, 2, 3)
     * ConstantAdapter<eigen_matrix_t<double, Eigen::Dynamic, 3>>(5., 2, 3)
     * ConstantAdapter<eigen_matrix_t<double, Eigen::Dynamic, Eigen::Dynamic>>(5., 2, 3)
     * \endcode
     */
#ifdef __cpp_concepts
    template<scalar_constant C, index_descriptor...Ds> requires std::constructible_from<MyConstant, C&&> and
      compatible_with_index_descriptors<PatternMatrix, Ds...>
#else
    template<typename C, typename...Ds, std::enable_if_t<scalar_constant<C> and (index_descriptor<Ds> and ...) and
      std::is_constructible_v<MyConstant, C&&> and compatible_with_index_descriptors<PatternMatrix, Ds...>, int> = 0>
#endif
    explicit constexpr ConstantAdapter(C&& c, Ds&&...ds) : my_constant {std::forward<C>(c)},
      my_dimensions {make_all_dimensions_tuple(std::forward<Ds>(ds)...)} {}

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
     * \overload
     * \brief Construct a ConstantAdapter, whose value is known at compile time, using only applicable <em>dynamic</em> index descriptors.
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
      (sizeof...(Ds) == number_of_dynamic_indices_v<PatternMatrix>) and scalar_constant<MyConstant, CompileTimeStatus::known>
#else
    template<typename...Ds, std::enable_if_t<(dynamic_index_descriptor<Ds> and ...) and
      (sizeof...(Ds) < std::tuple_size_v<MyDimensions>) and sizeof...(Ds) == number_of_dynamic_indices_v<PatternMatrix> and
      scalar_constant<MyConstant, CompileTimeStatus::known>, int> = 0>
#endif
    explicit constexpr ConstantAdapter(Ds&&...ds) : my_dimensions {make_dynamic_dimensions_tuple(std::forward<Ds>(ds)...)} {}


    /**
     * \overload
     * \brief Same as above, except specifying a \ref scalar_constant.
     * \details For example, the following construct a 2-by-3 constant matrix of value 5:
     * \code
     * ConstantAdapter<eigen_matrix_t<double, 2, dynamic_size>, 5>(std::integral_constant<int, 5>{}, 3) // Fixed rows and dynamic columns.
     * ConstantAdapter<eigen_matrix_t<double, 2, 3>>(5) // Fixed rows and columns.
     * ConstantAdapter<eigen_matrix_t<double, dynamic_size, 3>>(5, 2) // Dynamic rows and fixed columns.
     * ConstantAdapter<eigen_matrix_t<double, 2, dynamic_size>, 5>(5, 3) // Fixed rows and dynamic columns.
     * \endcode
     */
#ifdef __cpp_concepts
    template<scalar_constant C, dynamic_index_descriptor...Ds> requires
      (sizeof...(Ds) < std::tuple_size_v<MyDimensions>) and
      (sizeof...(Ds) == number_of_dynamic_indices_v<PatternMatrix>) and std::constructible_from<MyConstant, C&&>
#else
    template<typename C, typename...Ds, std::enable_if_t<scalar_constant<C> and (dynamic_index_descriptor<Ds> and ...) and
      (sizeof...(Ds) < std::tuple_size_v<MyDimensions>) and
      (sizeof...(Ds) == number_of_dynamic_indices_v<PatternMatrix>) and std::is_constructible_v<MyConstant, C&&>, int> = 0>
#endif
    explicit constexpr ConstantAdapter(C&& c, Ds&&...ds) : my_constant {std::forward<C>(c)},
      my_dimensions {make_dynamic_dimensions_tuple(std::forward<Ds>(ds)...)} {}

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
     * \overload
     * \brief Construct a ConstantMatrix from another \ref constant_matrix.
     */
#ifdef __cpp_concepts
    template<constant_matrix<Likelihood::definitely, CompileTimeStatus::any> Arg> requires
      (not std::derived_from<Arg, ConstantAdapter>) and maybe_has_same_shape_as<Arg, PatternMatrix> and
      std::constructible_from<MyConstant, constant_coefficient<Arg>>
#else
    template<typename Arg, std::enable_if_t<constant_matrix<Arg, Likelihood::definitely, CompileTimeStatus::any> and
      (not std::is_base_of_v<ConstantAdapter, Arg>) and maybe_has_same_shape_as<Arg, PatternMatrix> and
      std::is_constructible_v<MyConstant, constant_coefficient<Arg>>, int> = 0>
#endif
    constexpr ConstantAdapter(const Arg& arg) :
      my_constant {constant_coefficient {arg}}, my_dimensions {make_dimensions_tuple(arg)} {}


    /**
     * \overload
     * \brief Construct a ConstantAdapter based on a library object and a \ref scalar_constant known at compile time.
     * \details This is for use with the corresponding deduction guide.
     * The following constructs a 2-by-3 ConstantAdapter with constant 5 (known at compile time, or at runtime, respectively).
     * \code
     * ConstantAdapter {std::integral_constant<int, 5>{}, eigen_matrix_t<double, 2, 3>{})
     * ConstantAdapter {std::integral_constant<int, 5>{}, eigen_matrix_t<double, 2, Eigen::Dynamic>(2, 3))
     * ConstantAdapter {5, eigen_matrix_t<double, 2, 3>{})
     * ConstantAdapter {5, eigen_matrix_t<double, Eigen::Dynamic, 3>(2, 3))
     * \endcode
     */
#ifdef __cpp_concepts
    template<scalar_constant C, indexible Arg> requires
      maybe_has_same_shape_as<Arg, PatternMatrix> and std::constructible_from<MyConstant, C&&>
#else
    template<typename C, typename Arg, std::enable_if_t<scalar_constant<C> and indexible<Arg> and
      maybe_has_same_shape_as<Arg, PatternMatrix> and std::is_constructible_v<MyConstant, C&&>, int> = 0>
#endif
    constexpr ConstantAdapter(C&& c, const Arg& arg) :
      my_constant {std::forward<C>(c)}, my_dimensions {make_dimensions_tuple(arg)} {}


    /**
     * \brief Assign from another compatible \ref constant_matrix.
     */
#ifdef __cpp_concepts
    template<constant_matrix<Likelihood::definitely, CompileTimeStatus::any> Arg> requires
      (not std::derived_from<Arg, ConstantAdapter>) and maybe_has_same_shape_as<Arg, PatternMatrix> and
      std::assignable_from<MyConstant, constant_coefficient<Arg>>
#else
    template<typename Arg, std::enable_if_t<
      constant_matrix<Arg, Likelihood::definitely, CompileTimeStatus::any> and
      (not std::is_base_of_v<ConstantAdapter, Arg>) and maybe_has_same_shape_as<Arg, PatternMatrix> and
      std::is_assignable_v<MyConstant, constant_coefficient<Arg>>, int> = 0>
#endif
    constexpr auto& operator=(const Arg& arg)
    {
      if constexpr (not has_same_shape_as<Arg, PatternMatrix>) if (not get_index_descriptors_match(*this, arg))
        throw std::invalid_argument {"Argument to ConstantAdapter assignment operator has non-matching index descriptors."};
      my_constant = constant_coefficient {arg};
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
        return false;
      else if constexpr (constant_matrix<Arg, Likelihood::definitely, CompileTimeStatus::any>)
        return get_scalar_constant_value(constant_coefficient{arg}) == get_scalar_constant_value(my_constant) and get_index_descriptors_match(*this, arg);
      else
      {
        auto c = to_native_matrix<PatternMatrix>(*this);
        static_assert(not std::is_same_v<decltype(c), ConstantAdapter>,
          "interface::EquivalentDenseWritableMatrix<PatternMatrix>::to_native_matrix(*this) must define an object within the library of Arg");
        return std::move(c) == arg;
      }
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
      else if constexpr (constant_matrix<Arg, Likelihood::definitely, CompileTimeStatus::any>)
        return get_scalar_constant_value(constant_coefficient{arg}) == get_scalar_constant_value(c.get_scalar_constant()) and get_index_descriptors_match(arg, c);
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
    constexpr scalar_type auto
#else
    template<typename...Is, std::enable_if_t<
      (index_value<Is> and ...) and (sizeof...(Is) <= max_indices_of_v<PatternMatrix>), int> = 0>
    constexpr auto
#endif
    operator()(Is...is) const
    {
      return get_scalar_constant_value(my_constant);
    }


    /**
     * \brief Element accessor (single index).
     * \note Does not do any runtime bounds checking.
     * \param is The index
     * \return The element corresponding to the indices (always the constant).
     */
#ifdef __cpp_concepts
    template<index_value Is>
    constexpr scalar_type auto
#else
    template<typename Is, std::enable_if_t<index_value<Is>, int> = 0>
    constexpr auto
#endif
    operator[](Is is) const
    {
      return get_scalar_constant_value(my_constant);
    }


    /**
     * \brief Get the \ref scalar_constant associated with this object.
     */
#ifdef __cpp_concepts
    constexpr scalar_constant auto
#else
    constexpr auto
#endif
    get_scalar_constant() const
    {
      return my_constant;
    }

  protected:

    MyConstant my_constant;

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
  template<scalar_constant C, indexible Arg>
#else
  template<typename C, typename Arg, std::enable_if_t<scalar_constant<C> and indexible<Arg>, int> = 0>
#endif
  ConstantAdapter(const C&, const Arg&) -> ConstantAdapter<Arg, C>;

#ifdef __cpp_concepts
  template<constant_matrix<Likelihood::definitely, CompileTimeStatus::any> Arg> requires (not constant_adapter<Arg>)
#else
  template<typename Arg, std::enable_if_t<constant_matrix<Arg, Likelihood::definitely, CompileTimeStatus::any> and
    (not constant_adapter<Arg>), int> = 0>
#endif
  ConstantAdapter(const Arg&) -> ConstantAdapter<Arg, constant_coefficient<Arg>>;


  // ------------ //
  //  Interfaces  //
  // ------------ //

  namespace interface
  {
    template<typename PatternMatrix, typename Scalar, auto...constant>
    struct IndexibleObjectTraits<ConstantAdapter<PatternMatrix, Scalar, constant...>>
    {
      static constexpr std::size_t max_indices = max_indices_of_v<PatternMatrix>;
      using scalar_type = std::decay_t<decltype(get_scalar_constant_value(
        std::declval<ConstantAdapter<PatternMatrix, Scalar, constant...>>().get_scalar_constant()))>;
    };


    template<typename PatternMatrix, typename Scalar, auto...constant, std::size_t N>
    struct IndexTraits<ConstantAdapter<PatternMatrix, Scalar, constant...>, N>
    {
      static constexpr std::size_t dimension = index_dimension_of_v<PatternMatrix, N>;

      template<typename Arg>
      static constexpr std::size_t dimension_at_runtime(const Arg& arg)
      {
        return get_dimension_size_of(std::get<N>(arg.my_dimensions));
      }
    };


    template<typename PatternMatrix, typename Scalar, auto...constant, std::size_t N>
    struct CoordinateSystemTraits<ConstantAdapter<PatternMatrix, Scalar, constant...>, N>
    {
      using coordinate_system_types = coefficient_types_of_t<PatternMatrix, N>;

      template<typename Arg>
      static constexpr auto coordinate_system_types_at_runtime(Arg&& arg)
      {
        return std::get<N>(std::forward<Arg>(arg).my_dimensions);
      }
    };


    template<typename PatternMatrix, typename Scalar, auto...constant, typename...I>
#ifdef __cpp_concepts
    struct GetElement<ConstantAdapter<PatternMatrix, Scalar, constant...>, I...>
#else
    struct GetElement<ConstantAdapter<PatternMatrix, Scalar, constant...>, void, I...>
#endif
    {
      template<typename Arg>
      static constexpr auto get(Arg&& arg, I...) { return constant_coefficient_v<Arg>; }
    };


    // No SetElement defined  because ConstantAdapter is not writable.


    template<typename PatternMatrix, typename S, auto...constant, typename Scalar>
    struct EquivalentDenseWritableMatrix<ConstantAdapter<PatternMatrix, S, constant...>, Scalar>
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


    template<typename PatternMatrix, typename Scalar, auto...constant>
    struct Dependencies<ConstantAdapter<PatternMatrix, Scalar, constant...>>
    {
      static constexpr bool has_runtime_parameters = has_dynamic_dimensions<PatternMatrix>;
      using type = std::tuple<>;
    };


    template<typename PatternMatrix, typename S, auto...constant, typename Scalar>
    struct SingleConstantMatrixTraits<ConstantAdapter<PatternMatrix, S, constant...>, Scalar>
    {
      template<typename C, typename...D>
      static constexpr auto make_constant_matrix(C&& c, D&&...d)
      {
        return make_constant_matrix_like<PatternMatrix>(std::forward<C>(c), std::forward<D>(d)...);
      }
    };


    template<typename PatternMatrix, typename Scalar, auto...cs>
    struct SingleConstant<ConstantAdapter<PatternMatrix, Scalar, cs...>>
    {
      const ConstantAdapter<PatternMatrix, Scalar, cs...>& xpr;

      constexpr auto get_constant() { return xpr.get_scalar_constant(); }
    };


    template<typename PatternMatrix, typename S, auto...constant, typename Scalar>
    struct SingleConstantDiagonalMatrixTraits<ConstantAdapter<PatternMatrix, S, constant...>, Scalar>
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


    template<typename PatternMatrix, typename Scalar, auto...constant>
    struct ArrayOperations<ConstantAdapter<PatternMatrix, Scalar, constant...>>
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


    template<typename PatternMatrix, typename Scalar, auto...constant>
    struct LinearAlgebra<ConstantAdapter<PatternMatrix, Scalar, constant...>>
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

  template<typename PatternMatrix, typename Scalar, auto...constant>
  struct MatrixTraits<ConstantAdapter<PatternMatrix, Scalar, constant...>>
  {
  private:

    using Matrix = ConstantAdapter<PatternMatrix, Scalar, constant...>;

  public:

    template<typename Derived>
    using MatrixBaseFrom = typename MatrixTraits<std::decay_t<PatternMatrix>>::template MatrixBaseFrom<Derived>;

    template<TriangleType storage_triangle = TriangleType::diagonal>
    using SelfAdjointMatrixFrom = SelfAdjointMatrix<Matrix, storage_triangle>;

    template<TriangleType triangle_type = TriangleType::diagonal>
    using TriangularMatrixFrom = TriangularMatrix<Matrix, triangle_type>;

    template<std::size_t dim = row_dimension_of_v<PatternMatrix>>
    using DiagonalMatrixFrom = DiagonalMatrix<ConstantAdapter<
      untyped_dense_writable_matrix_t<PatternMatrix, Scalar, dim, 1>, Scalar, constant...>>;

  };


} // namespace OpenKalman


#endif //OPENKALMAN_CONSTANTADAPTER_HPP
